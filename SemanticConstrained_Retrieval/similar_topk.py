import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import faiss


class AddressRetriever:
    def __init__(self, args, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sbert = SentenceTransformer(args.sbert)

        # ✅ 文件路径
        self.model_path = os.path.join(args.graph_emb, "best_model.pth")
        self.emb_path = os.path.join(args.graph_emb, "graph_embeddings.pt")
        self.json_path = os.path.join(args.graph_emb, "community_embeddings.json")

        # ✅ 加载 JSON 数据
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        print(f"✅ dataset 加载完成，共 {len(self.dataset)} 条记录")

        # ✅ 加载 embedding
        self.graph_embeddings = torch.load(self.emb_path).cpu()
        self.dim = self.graph_embeddings.shape[1]
        print(f"✅ embedding 加载完成，维度={self.dim}")

        # ✅ 一致性检查（防止索引越界）
        if len(self.dataset) != self.graph_embeddings.shape[0]:
            raise ValueError(
                f"❌ 数据不一致：community_embeddings.json 行数={len(self.dataset)}，"
                f"graph_embeddings.pt 行数={self.graph_embeddings.shape[0]}。\n"
                f"请重新生成文件，确保 JSON 和 embedding 来自同一次训练。"
            )

        # ✅ 构建 FAISS 索引
        self.index = faiss.IndexFlatIP(self.dim)
        emb_np = self.graph_embeddings.numpy().astype("float32")
        faiss.normalize_L2(emb_np)
        self.index.add(emb_np)
        print(f"✅ FAISS 索引构建完成，索引大小={self.index.ntotal}")

        # ✅ 加载 Projection
        ckpt = torch.load(self.model_path, map_location=self.device)
        self.projection = nn.Linear(768, self.dim).to(self.device)
        self.projection.load_state_dict(ckpt['proj'])
        self.projection.eval()
        print(f"✅ Projection 加载完成")

        # ✅ 缓存所有 district 和 street
        self.name = list({item['name'] for item in self.dataset})
        # self.district_set = list({item['district'] for item in self.dataset})
        # self.street_set = list({item['township'] for item in self.dataset})

        print(f"✅ 模型和索引加载完成")

    def _extract_candidates(self, address):
        matched_name = [d for d in self.name if d in address]
        # matched_districts = [d for d in self.district_set if d in address]
        # matched_streets = [s for s in self.street_set if s in address]
        # return matched_districts, matched_streets
        return matched_name

    def search(self, query_text, top_k=5):
        # ✅ 自动解析可能的 district 和 street
        # matched_districts, matched_streets = self._extract_candidates(query_text)
        matched_name = self._extract_candidates(query_text)

        # ✅ 编码查询
        query_emb = self.sbert.encode(query_text, convert_to_tensor=True)
        query_emb = self.projection(query_emb.clone().detach().unsqueeze(0))
        query_emb = F.normalize(query_emb, p=2, dim=-1).detach().cpu().numpy().astype('float32')
        faiss.normalize_L2(query_emb)

        # ✅ 初步检索
        D, I = self.index.search(query_emb, top_k * 100)  # 先多取，再加权过滤
        results = []
        for idx, score in zip(I[0], D[0]):
            item = self.dataset[idx]
            final_score = float(score)
            # # ✅ 匹配区加权
            # if item['district'] in matched_districts:
            #     final_score += 0.1
            # # ✅ 匹配街道加权
            # if item['township'] in matched_streets:
            #     final_score += 0.1
            if item['name'] in matched_name:
                final_score += 0.3
                print(item['name'])
            results.append({
                'name': item['name'],
                'prov': item['prov'],
                'district': item['district'],
                'township': item['township'],
                'score': final_score
            })

        # ✅ 根据加权后的分数排序
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        return results
