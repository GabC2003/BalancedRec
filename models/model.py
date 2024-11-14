import torch.nn as nn
from tqdm import tqdm
import pdb
import torch.nn.functional as F
import torch
from dgl.nn import SAGEConv
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GATConv
from dgl.nn import GraphConv
from torch_kmeans import KMeans
class Proposed_model(nn.Module):
    def __init__(self, args, graph, item_graph):
        super().__init__()
        self.args = args
        self.hid_dim = args.embed_size
        self.layer_num = args.layers

        self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))
        self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('game').shape[0], self.hid_dim))

        # self.user_embedding = torch.nn.Parameter(torch.load('./baselines/user_embedding.pt'))
        # self.item_embedding = torch.nn.Parameter(torch.load('./baselines/item_embedding.pt'))

        self.item_conv = SAGEConv(self.hid_dim, self.hid_dim, 'mean')
        self.social_GAT = GATConv(self.hid_dim, self.hid_dim, num_heads = 1, allow_zero_in_degree = True)
        self.social_conv = SAGEConv(self.hid_dim, self.hid_dim, 'mean')
        self.linear = torch.nn.Linear(3 * self.hid_dim, self.hid_dim)
        self.kmeans = KMeans(n_clusters=10, verbose=False,init_method='k-means++')
        self.build_model(item_graph)
        self.user_game_graph = graph.edge_type_subgraph(['play'])

    def build_layer(self, idx, graph):
        if idx == 0:
            input_dim = graph.ndata['h'].shape[1]
        else:
            input_dim = self.hid_dim
        dic = {
            rel: GraphConv(input_dim, self.hid_dim, weight = True, bias = False)
            for rel in graph.etypes
        }
        return dglnn.HeteroGraphConv(dic, aggregate = 'mean')

    def build_model(self, graph):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx, graph)
            self.layers.append(h2h)

    def calculate_h_user_diversity(self, clustered_labels, clustered_centers):
        self.user_game_graph.nodes['game'].data['cluster'] = clustered_labels
        
        user_embeds = self.user_embedding.unsqueeze(1)
        centers = clustered_centers.unsqueeze(0)
        distances = torch.cdist(user_embeds, centers).squeeze(1)
        
        weights = F.softmax(distances, dim=1)
        h_user_diversity = torch.matmul(weights, clustered_centers)
        
        return h_user_diversity
    
    def calculate_div(self, user_game_graph, user_embed, clustered_centers, clustered_labels):
        user_game_graph = user_game_graph.local_var()  
        user_game_graph.nodes['game'].data['cluster'] = clustered_labels
        user_game_graph.nodes['game'].data['cluster_center'] = clustered_centers[clustered_labels]
        user_game_graph.nodes['user'].data['h'] = user_embed

        game_embed = self.item_embedding
        game_to_cluster_distances = torch.norm(game_embed - clustered_centers[clustered_labels], dim=1)

        user_indices, game_indices = user_game_graph.edges(etype='play')
        user_embeds = user_embed[user_indices]

        cluster_centers = clustered_centers[clustered_labels[game_indices]]
        
        diff = user_embeds - cluster_centers
        sq_diff = torch.sum(diff ** 2, dim=1)

        num_users = user_embed.shape[0]
        total_sq_diff = torch.zeros(num_users, device=user_embed.device)
        num_unique_clusters = torch.zeros(num_users, device=user_embed.device)
        
        total_sq_diff.index_add_(0, user_indices, sq_diff)
        num_unique_clusters.index_add_(0, user_indices, torch.ones_like(user_indices, dtype=torch.float))
        
        diversity_losses = total_sq_diff / (num_unique_clusters + 1e-10) 
        
        return diversity_losses, game_to_cluster_distances
    
    def forward(self, graph, item_graph, social_graph):

        h_game = item_graph.ndata['h']
        for layer in self.layers:
            h_game = layer(item_graph, {'game': h_game})['game']

        graph_game2user = dgl.edge_type_subgraph(graph, ['played by'])

        weight = graph.edata['weight'][('game', 'played by', 'user')]
        h_user_aggregate = self.item_conv(graph_game2user, (h_game, self.user_embedding), edge_weight = weight)

        _, social_weight = self.social_GAT(social_graph, h_user_aggregate, get_attention = True)
        social_weight = social_weight.sum(1)
        h_user_social = self.social_conv(social_graph, self.user_embedding, edge_weight = social_weight)
        
        with torch.no_grad():
            game_embed_expanded = self.item_embedding.unsqueeze(0)
            clustered_result = self.kmeans(game_embed_expanded)
            clustered_labels = clustered_result.labels.squeeze(0)
            clustered_centers = clustered_result.centers.squeeze(0)

        h_user_diversity = self.calculate_h_user_diversity(clustered_labels,clustered_centers)
        

        user_embed = (1 - self.args.social_g - self.args.item_g) * (self.user_embedding + self.args.diversity_g * h_user_diversity) + self.args.item_g * h_user_aggregate + self.args.social_g * h_user_social
        
        #针对前threshold_g的用户做个性化处理，取消计算他们的多样性embedding和多样性损失
        diversity_losses, min_distances = self.calculate_div(self.user_game_graph, user_embed, clustered_centers, clustered_labels)
        num_users = diversity_losses.shape[0]
        threshold_index = int(num_users * self.args.threshold_g)
        sorted_losses, _ = torch.sort(diversity_losses,descending=True)
        threshold = sorted_losses[threshold_index]

        mask = diversity_losses > threshold
        masked_diversity_losses = diversity_losses[~mask]

        filtered_div_loss = masked_diversity_losses.mean()

        h_user_diversity = h_user_diversity * (~mask).float().unsqueeze(1)

        mask_float = mask.float().unsqueeze(1)

        user_embed = user_embed - (h_user_diversity * mask_float)
        
        self.user_game_graph.nodes['user'].data['h'] = user_embed

        return {"user": user_embed, "game": self.item_embedding}, filtered_div_loss,min_distances,mask

### 个性化怎么设置指标/div_loss的计算方式/user_div_embedding嵌入的时机
