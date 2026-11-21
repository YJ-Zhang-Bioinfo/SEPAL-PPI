#!/usr/bin/env python3
"""
预计算多模态特征并存储到 LMDB
消除逐样本循环瓶颈，实现批量化GPU加速

用法:
    python mutifeature_tools/precompute_multimodal_lmdb.py \
        --config config/model_unit/preprocessing/feature_concat.yaml --schema BPP_RSA_Q3_1.0 \
        --feature-folder mutifeature/Strings_plant50 --fasta dataset/Strings_plant50/protein.fasta \
        --visible-cap 1000 --output /path/to/new/multimodal_features.lmdb
"""

import argparse
import json
import logging
import lmdb
import numpy as np
import tempfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple
from Bio import SeqIO

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BPP_SCHEMA = 'BPP_RSA_Q3_1.0'


def parse_q3(value):
    """Decode one H/E/C triple per residue, including the lite writer's row string."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('['):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = json.loads('[' + value + ']')  # [1,0,0],[0,1,0]
        else:
            mapping = {'H': (1, 0, 0), 'E': (0, 1, 0), 'C': (0, 0, 1)}
            value = [mapping[c] for c in value]
    rows = np.asarray(value, dtype=np.float32)
    if rows.shape == (3,):  # a single unwrapped triple
        rows = rows[None, :]
    if (rows.ndim != 2 or rows.shape[1] != 3 or not len(rows)
            or not ((rows == 0) | (rows == 1)).all() or not (rows.sum(axis=1) == 1).all()):
        raise ValueError('expected non-empty one-hot Q3 rows in H/E/C order')
    return rows


class MultimodalFeaturePreprocessor:
    """多模态特征预处理器：JSON → LMDB"""
    
    def __init__(self, config_path: str, feature_folder=None, fasta=None, visible_cap=None, schema=None):
        with Path(config_path).open(encoding='utf-8') as handle:
            self.config = yaml.safe_load(handle)
        processing = self.config['data_processing']
        self.feature_folder = feature_folder or processing.get('all_feature_folder')
        self.fasta = fasta or processing.get('protein_fasta_path')
        self.schema = schema or processing.get('required_lmdb_schema')
        self.visible_cap = visible_cap if visible_cap is not None else processing.get('visible_cap')
        if not self.feature_folder or '${' in str(self.feature_folder):
            raise ValueError('Specify the raw JSON directory with --feature-folder')
        if self.schema not in (None, BPP_SCHEMA):
            raise ValueError(f'Unsupported required_lmdb_schema: {self.schema}')
        if self.visible_cap is not None and (not self.schema or not isinstance(self.visible_cap, int) or self.visible_cap < 1):
            raise ValueError('visible_cap must be a positive integer for the BPP schema')
        self.feature_files = self.config['feature_files']
        
        # 只处理启用的特征
        self.enabled_features = {
            name: cfg for name, cfg in self.feature_files.items()
            if cfg.get('enabled', False)
        }
        
        logger.info(f"启用的特征: {list(self.enabled_features.keys())}")
        
        # 计算特征维度和顺序
        self.feature_order = list(self.enabled_features)  # same order as FeatureConcatUnit
        if not self.feature_order:
            raise ValueError('No enabled feature files')
        self.feature_dims = []
        self.total_dim = 0
        
        for name in self.feature_order:
            cfg = self.enabled_features[name]
            if cfg.get('feature_type') == 'embedding':
                # embedding特征存储ID，维度为1
                dim = 1
            else:
                dim = cfg.get('raw_dim', cfg.get('input_dim', cfg.get('feature_dim', 1)))
            self.feature_dims.append(dim)
            self.total_dim += dim
        
        logger.info(f"特征维度: {dict(zip(self.feature_order, self.feature_dims))}")
        logger.info(f"总维度: {self.total_dim}")
        if self.schema and (self.feature_order != ['sasa_features', 'secondary_structure_features']
                            or self.feature_dims != [1, 3]
                            or any(cfg.get('feature_type') == 'embedding' for cfg in self.enabled_features.values())):
            raise ValueError(f'{self.schema} requires RSA and Q3 in that order with dimensions [1, 3]')
    
    def load_feature_data(self, feature_name: str) -> Dict[str, Any]:
        """加载单个特征文件"""
        cfg = self.enabled_features[feature_name]
        file_path = cfg.get('file_path')
        
        if not file_path:
            return {}
        
        full_path = Path(self.feature_folder) / file_path
        if not full_path.exists():
            if self.schema:
                raise FileNotFoundError(full_path)
            logger.warning(f"特征文件不存在: {full_path}")
            return {}
        
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # 转换为 {protein_id: data} 格式
        feature_dict = {}
        for item in data:
            protein_id = item.get('protein_id') or item.get('Protein_ID')
            if self.schema:
                if not isinstance(protein_id, str) or not protein_id.strip():
                    raise ValueError(f'Missing protein ID in {full_path}')
                protein_id = protein_id.upper()  # source IDs may differ in case from FASTA
                if protein_id in feature_dict:
                    raise ValueError(f'Duplicate protein ID {protein_id} in {full_path}')
            if protein_id:
                feature_dict[protein_id] = item
        
        logger.info(f"加载 {feature_name}: {len(feature_dict)} 条")
        return feature_dict
    
    def extract_feature_array(self, item: Dict, feature_name: str, config: Dict) -> np.ndarray:
        """从JSON项提取特征数组"""
        
        # embedding特征：只存储ID
        if config.get('feature_type') == 'embedding':
            if feature_name == 'protein_length':
                # 蛋白长度需要从序列长度计算
                return None  # 稍后处理
            
            # 其他embedding特征：提取ID序列
            data_field = feature_name.replace('_features', '').replace('_', '')
            if feature_name == 's1ssttoken':
                data_field = 'prosst_structure_tokens'
            elif feature_name == 'hmm_pfam_id':
                data_field = 'hmm_pfam_id'
            
            id_str = item.get(data_field, '')
            if not id_str:
                return np.array([], dtype=np.int32)
            
            # embedding特征：只存储ID
            ids = [int(x) for x in id_str.split(',') if x.strip()]
            return np.array(ids, dtype=np.float32).reshape(-1, 1)  # 统一为float32
        
        # 普通特征：提取浮点数
        if feature_name == 'secondary_structure_features':
            return parse_q3(item['secondary_structure'])
        
        # 其他数值特征
        data_field = feature_name.replace('_features', '').replace('_', '')
        if 'hydrophobicity' in feature_name:
            data_field = 'hydrophobicity'
        elif 'sasa' in feature_name:
            data_field = 'sasa'
        elif 'pqr' in feature_name:
            data_field = 'pqr'
        
        value_str = item.get(data_field, item.get(feature_name, ''))
        if not value_str:
            return np.array([], dtype=np.float32)
        
        values = [float(x) for x in str(value_str).split(',') if x.strip()]
        feature_dim = config.get('raw_dim', config.get('feature_dim', 1))
        
        if feature_dim == 1:
            arr = np.array(values, dtype=np.float32).reshape(-1, 1)
        else:
            usable = (len(values) // feature_dim) * feature_dim
            arr = np.array(values[:usable], dtype=np.float32).reshape(-1, feature_dim)
        
        return arr
    
    def bpp_lengths(self):
        if not self.fasta or not Path(self.fasta).is_file():
            raise ValueError(f'BPP requires the target FASTA (--fasta); got {self.fasta}')
        lengths, seen = {}, set()
        with Path(self.fasta).open(encoding='utf-8') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                pid = record.id
                if not pid or not len(record) or pid.startswith('__') or pid.upper() in seen:
                    raise ValueError(f'Empty sequence, reserved or duplicate FASTA ID: {pid}')
                seen.add(pid.upper())
                lengths[pid] = len(record)
        if not lengths:
            raise ValueError('Empty target FASTA')
        return lengths

    def bpp_array(self, pid, length, features):
        raw = features['sasa_features'][pid.upper()]['sasa']
        rsa = np.asarray(raw.split(',') if isinstance(raw, str) else raw, dtype=np.float32)
        q3 = parse_q3(features['secondary_structure_features'][pid.upper()]['secondary_structure'])
        n = min(length, self.visible_cap) if self.visible_cap else length
        if (rsa.ndim != 1 or min(len(rsa), len(q3)) < n
                or (self.visible_cap is None and (len(rsa) != n or len(q3) != n))):
            raise ValueError(f'FASTA length={length}, required={n}, RSA shape={rsa.shape}, Q3 rows={len(q3)}')
        rsa = rsa[:n]
        if not ((rsa >= 0) & (rsa <= 1)).all():
            raise ValueError('RSA must be finite and in [0, 1]')
        return np.column_stack((rsa, q3[:n]))
    
    def process_protein(self, protein_id: str, all_features: Dict[str, Dict]) -> Tuple[np.ndarray, Dict]:
        """处理单个蛋白的所有特征"""
        features_list = []
        metadata = {'protein_id': protein_id, 'feature_order': self.feature_order}
        
        for feature_name in self.feature_order:
            cfg = self.enabled_features[feature_name]
            
            if feature_name not in all_features:
                # 特征文件未加载
                dim = self.feature_dims[self.feature_order.index(feature_name)]
                arr = np.zeros((0, dim), dtype=np.float32)
            elif protein_id not in all_features[feature_name]:
                # 蛋白在该特征文件中缺失
                dim = self.feature_dims[self.feature_order.index(feature_name)]
                arr = np.zeros((0, dim), dtype=np.float32)
            else:
                item = all_features[feature_name][protein_id]
                arr = self.extract_feature_array(item, feature_name, cfg)
                
                if arr is None or len(arr) == 0:
                    dim = self.feature_dims[self.feature_order.index(feature_name)]
                    arr = np.zeros((0, dim), dtype=np.float32)
            
            features_list.append(arr)
        
        # 找到最大序列长度
        max_len = max((arr.shape[0] for arr in features_list if arr.shape[0] > 0), default=0)
        metadata['seq_len'] = max_len
        
        # Padding 所有特征到相同长度
        padded_features = []
        for idx, (arr, dim) in enumerate(zip(features_list, self.feature_dims)):
            feature_name = self.feature_order[idx]
            cfg = self.enabled_features[feature_name]
            is_embedding = cfg.get('feature_type') == 'embedding'
            
            if arr.shape[0] == 0:
                # 空特征：填充零
                dtype = np.int32 if is_embedding else np.float32
                padded = np.zeros((max_len, dim), dtype=dtype)
            elif arr.shape[0] < max_len:
                # Padding
                pad_len = max_len - arr.shape[0]
                padded = np.vstack([arr, np.zeros((pad_len, dim), dtype=arr.dtype)])
            else:
                # Truncate
                padded = arr[:max_len]
            
            # 统一转换为float32（包括embedding ID）
            padded_features.append(padded.astype(np.float32))
        
        # 拼接: [seq_len, total_dim]
        combined = np.concatenate(padded_features, axis=-1)
        return combined, metadata
    
    def build_lmdb(self, output_path: str, map_size: int = 50 * 1024**3, force=False):
        """构建LMDB数据库"""
        
        # 加载所有特征文件
        logger.info("加载所有特征文件...")
        all_features = {}
        for feature_name in tqdm(self.feature_order, desc="Loading features"):
            all_features[feature_name] = self.load_feature_data(feature_name)
        
        # 获取所有蛋白ID
        lengths = self.bpp_lengths() if self.schema else {}
        protein_ids = lengths.keys() if self.schema else all_features[self.feature_order[0]].keys()
        if not protein_ids:
            raise ValueError('No proteins to write')
        logger.info(f"总共 {len(protein_ids)} 个蛋白")
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        global_meta = dict(feature_order=self.feature_order, feature_dims=self.feature_dims,
                           total_dim=self.total_dim, num_proteins=len(protein_ids))
        if self.schema:
            global_meta.update(schema_version=self.schema, visible_cap=self.visible_cap)

        # A first-time reader must never see a partially created store. For an
        # explicit offline --force rebuild, replace the keyspace in one transaction.
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            existing = output.exists()
            if existing and not force:
                raise FileExistsError(output)
            if existing and not (output / 'data.mdb').is_file():
                raise ValueError(f'Refusing to overwrite a non-LMDB directory: {output}')
            dest = output if existing else Path(tmp) / 'store'
            with lmdb.open(str(dest), map_size=map_size) as env, env.begin(write=True) as txn:
                if existing:
                    if not txn.get(b'__global_meta__'):
                        raise ValueError(f'Refusing to overwrite an unrecognized LMDB: {output}')
                    txn.drop(env.open_db(None, txn=txn), delete=False)
                for protein_id in tqdm(protein_ids, desc='Processing proteins'):
                    try:
                        if self.schema:
                            combined = self.bpp_array(protein_id, lengths[protein_id], all_features)
                            metadata = {'seq_len': len(combined)}
                        else:
                            combined, metadata = self.process_protein(protein_id, all_features)
                    except (TypeError, KeyError, ValueError) as exc:
                        raise ValueError(f'Invalid features for {protein_id}: {exc}') from exc
                    txn.put(protein_id.encode(), combined.tobytes())
                    txn.put(f'__meta_{protein_id}'.encode(), json.dumps(metadata).encode())
                txn.put(b'__global_meta__', json.dumps(global_meta).encode())
            if not existing:
                dest.rename(output)
        logger.info(f"✓ LMDB 已保存到: {output_path}")
        logger.info(f"✓ data.mdb: {(output / 'data.mdb').stat().st_size / 1024**2:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="预计算多模态特征到LMDB")
    parser.add_argument('--config', required=True, help='特征配置文件路径')
    parser.add_argument('--feature-folder', help='输入 JSON 目录，覆盖 all_feature_folder')
    parser.add_argument('--fasta', help='BPP 目标蛋白 FASTA，覆盖 protein_fasta_path')
    parser.add_argument('--schema', choices=[BPP_SCHEMA], help='Override data_processing.required_lmdb_schema')
    parser.add_argument('--visible-cap', type=int, help='BPP 模型可见残基数；默认校验全长')
    parser.add_argument('--output', default=None, help='输出LMDB路径（默认：特征文件夹下的multimodal_features.lmdb）')
    parser.add_argument('--map-size', type=int, default=50, help='LMDB大小(GB)')
    parser.add_argument('--force', action='store_true', help='强制重新生成LMDB（即使已存在）')
    
    args = parser.parse_args()
    
    preprocessor = MultimodalFeaturePreprocessor(args.config, args.feature_folder, args.fasta, args.visible_cap, args.schema)
    
    # 自动确定输出路径
    if args.output is None:
        output_path = str(Path(preprocessor.feature_folder) / 'multimodal_features.lmdb')
    else:
        output_path = args.output
    
    # 检查是否已存在
    if Path(output_path).exists() and not args.force:
        if preprocessor.schema:
            with lmdb.open(output_path, readonly=True, lock=False, create=False) as env, env.begin() as txn:
                meta = json.loads(txn.get(b'__global_meta__') or b'{}')
                if (meta.get('schema_version') != preprocessor.schema
                        or meta.get('feature_order') != preprocessor.feature_order
                        or meta.get('feature_dims') != preprocessor.feature_dims):
                    parser.error(f'Existing LMDB is incompatible: {output_path}. Choose a new --output or rebuild offline with --force.')
        logger.warning(f"LMDB已存在: {output_path}")
        logger.warning("使用 --force 强制重新生成")
        return
    
    preprocessor.build_lmdb(output_path, map_size=args.map_size * 1024**3, force=args.force)


if __name__ == '__main__':
    main()
