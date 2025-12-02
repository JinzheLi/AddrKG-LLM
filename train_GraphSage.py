import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import SAGEConv
from pre_data.data_load import AddressDataLoader, AddressDataset
import faiss

sbert_path = "model/text2vec_base_chinese_paraphrase"

#################################################
# 多视图图构建函数
#################################################
def build_multi_graphs(dataset):
    nodes = {}
    node_type = {}
    node_id = 0
    edges_full = []
    edges_mid = []
    edges_comm = []


    def add_node(name, type_):
        nonlocal node_id
        if name not in nodes:
            nodes[name] = node_id
            node_type[node_id] = type_
            node_id += 1
        return nodes[name]

    for row in dataset:
        p = add_node(row['prov'], 'prov')
        d = add_node(row['district'], 'district')
        s = add_node(row['township'], 'township')
        c = add_node(row['name'], 'community')

        # 完整结构图
        edges_full += [(p, d), (d, s), (s, c)]

        # 街道+小区图
        edges_mid += [(s, c)]

        # 小区图（自连接）
        edges_comm += [(c, c)]

    to_tensor = lambda edges: torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).t().contiguous()
    return (
        nodes, node_type,
        to_tensor(edges_full),
        to_tensor(edges_mid),
        to_tensor(edges_comm)
    )

#################################################
# 模型
#################################################
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.layers.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

#################################################
# InfoNCE
#################################################
def info_nce_loss(query, positive, temperature=0.07):
    query = F.normalize(query, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    logits = torch.matmul(query, positive.T) / temperature
    labels = torch.arange(len(query)).to(query.device)
    return F.cross_entropy(logits, labels)

#################################################
# 训练流程
#################################################
def train_and_save_embeddings(dataset, args):
    # ✅ 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.save_path, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"训练结果将保存到: {save_dir}")

    model_path = os.path.join(save_dir, "best_model.pth")
    json_path = os.path.join(save_dir, "community_embeddings.json")
    tensor_path = os.path.join(save_dir, "graph_embeddings.pt")

    sbert = SentenceTransformer(sbert_path)
    nodes, node_type, edge_full, edge_mid, edge_comm = build_multi_graphs(dataset)

    node_texts = list(nodes.keys())
    node_features = sbert.encode(node_texts, convert_to_tensor=True).clone().detach().requires_grad_(True)

    # ✅ 构建两个视图
    addr_texts_full = [f"{row['prov']}{row['district']}{row['township']}{row['name']}" for row in dataset]
    addr_texts_name = [row['name'] for row in dataset]
    addr_embeddings_full = sbert.encode(addr_texts_full, convert_to_tensor=True).clone().detach()
    addr_embeddings_name = sbert.encode(addr_texts_name, convert_to_tensor=True).clone().detach()

    # ✅ Hard Negatives：仅使用 SBERT 向量找 hard negatives 的 index
    with torch.no_grad():
        sbert_vecs = sbert.encode(addr_texts_name, convert_to_tensor=True)
        sbert_vecs = F.normalize(sbert_vecs, p=2, dim=-1)
        sim_matrix = torch.matmul(sbert_vecs, sbert_vecs.T)
        # 去掉对角线自身最大相似度
        sim_matrix.fill_diagonal_(-1)
        hard_neg_index = torch.topk(sim_matrix, k=1, dim=1).indices.squeeze(1)  # 每行取最相似的另一个


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    node_features = node_features.to(device)
    addr_embeddings_full = addr_embeddings_full.to(device)
    addr_embeddings_name = addr_embeddings_name.to(device)
    edge_full, edge_mid, edge_comm = edge_full.to(device), edge_mid.to(device), edge_comm.to(device)
    hard_neg_index = hard_neg_index.to(device)

    model = GraphSAGE(node_features.shape[1], args.hidden_dim, args.out_dim, args.num_layers).to(device)
    projection = nn.Linear(addr_embeddings_full.shape[1], args.out_dim).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    best_loss = float('inf')
    wait = 0

    print("=" * 10, f"开始训练，共 {args.epochs} 轮", "=" * 10)
    for epoch in range(args.epochs):
        model.train()
        projection.train()
        optimizer.zero_grad()

        # GNN encode 三种图结构
        emb_full = model(node_features, edge_full)
        emb_mid = model(node_features, edge_mid)
        emb_comm = model(node_features, edge_comm)

        addr_proj_full = projection(addr_embeddings_full)  # full地址query
        addr_proj_name = projection(addr_embeddings_name)  # 小区名query

        pos_idx = [nodes[row['name']] for row in dataset]
        pos_emb_full = emb_full[pos_idx]
        pos_emb_mid = emb_mid[pos_idx]
        pos_emb_comm = emb_comm[pos_idx]

        # hard negative embedding
        neg_emb_full = emb_full[hard_neg_index]

        # ✅ 双正样本 InfoNCE Loss（平均）
        loss_full = (
                            info_nce_loss(addr_proj_full, pos_emb_full) +
                            info_nce_loss(addr_proj_full, pos_emb_mid) +
                            info_nce_loss(addr_proj_full, pos_emb_comm)
                    ) / 3

        loss_name = (
                            info_nce_loss(addr_proj_name, pos_emb_full) +
                            info_nce_loss(addr_proj_name, pos_emb_mid) +
                            info_nce_loss(addr_proj_name, pos_emb_comm)
                    ) / 3

        # ✅ Hard negative 损失（InfoNCE中的）
        query = F.normalize(addr_proj_full, p=2, dim=-1)
        positive = F.normalize(pos_emb_full, p=2, dim=-1)
        negative = F.normalize(neg_emb_full, p=2, dim=-1)
        # 拼接 [正样本 | 负样本]
        pos_plus_neg = torch.cat([positive, negative], dim=0)
        labels = torch.arange(len(query)).to(device)
        logits = torch.matmul(query, pos_plus_neg.T) / 0.07
        loss_hard = F.cross_entropy(logits, labels)

        loss = (loss_full + loss_name) / 2 + 0.5 * loss_hard  # 可调权重

        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({'model': model.state_dict(), 'proj': projection.state_dict()}, model_path)
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            print(f"早停触发，最佳 Loss: {best_loss:.4f}")
            break

    # 导出社区嵌入
    model.eval()
    with torch.no_grad():
        final_emb = model(node_features, edge_full)
        final_emb = F.normalize(final_emb, p=2, dim=-1)

    community_emb = [final_emb[nodes[row['name']]].cpu() for row in dataset]
    torch.save(torch.stack(community_emb), tensor_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{k: row[k] for k in ['prov', 'district', 'township', 'name']} for row in dataset], f,
                  ensure_ascii=False, indent=2)

    print(f"✅ 训练完成，embedding 行数={len(community_emb)}")
    return torch.stack(community_emb), nodes, dataset, projection, sbert


#################################################
# 检索
#################################################
def build_faiss_index(embeddings, dim=128):
    arr = embeddings.cpu().numpy().astype('float32')
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(dim)
    index.add(arr)
    return index

def search_address(query, sbert, projection, index, dataset, top_k=20):
    with torch.no_grad():
        q = sbert.encode(query, convert_to_tensor=True)
        q = projection(q.unsqueeze(0))
        q = F.normalize(q, p=2, dim=-1).cpu().numpy().astype('float32')
        faiss.normalize_L2(q)
    D, I = index.search(q, top_k)
    return [{**dataset[i], 'score': float(s)} for i, s in zip(I[0], D[0])]

#################################################
# 主入口
#################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="save/graph_emb")
    parser.add_argument("--data_path", type=str, default="data/Beijing_community.csv")
    args = parser.parse_args()

    print("=" * 50)
    print("训练启动，参数配置如下：")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 50)

    df = AddressDataLoader(args.data_path).load()
    dataset = AddressDataset(df)

    final_emb, nodes, dataset, projection, sbert = train_and_save_embeddings(dataset, args)
    index = build_faiss_index(final_emb)
    print("Top-K 示例：", search_address("海淀区西三旗龙乡小区", sbert, projection, index, dataset))

"""
python train_GraphSage.py  --data_path data/Beijing_community.csv   --epochs 1000  --num_layers 2 
"""