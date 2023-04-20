import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torchvision.models.video as models

from torch_geometric.nn import PointTransformerConv, MLP
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected



# This is tha main model that initializes the rest of the models
class MetaFormerGNN(nn.Module):
    def __init__(self, config):
        super(MetaFormerGNN, self).__init__()

        self.sub_models = config["sub_models"]
        self.pc = None
        self.backbone = None
        self.gnn = None
        for sub_model in self.sub_models:
            if sub_model == "backbone":
                self.backbone = ResNet3DBackbone(**config[sub_model])
                self.backbone.eval()
            elif sub_model == "pc":
                self.pc = PointCloud(**config[sub_model])
            elif sub_model == "gnn":
                self.gnn = GNNClassifier(**config[sub_model])
            else:
                raise NotImplementedError()
    
    def forward(self, x):
        # x: (batch_size, num_frames, 1, height, width)
        # edge_index: (2, num_edges)
        out_dict = {}
        vid = x["vid"]
        metadata_pc = x.get("metadata", None)
        pc_edge_index = x.get("pc_edge_index", None)
        pc_features = x.get("pc_features", None)
        mask = x['mask']
        batch_size = vid.shape[0]
        mask_reshaped = mask.view(batch_size, 1, -1)
        real_frames = []
        for b in range(mask_reshaped.shape[0]):
            for v in range(mask_reshaped.shape[1]):
                real_frames.append(torch.where(mask_reshaped[b, v])[0])

        # Create edges between adjacent real frames
        edges = []
        for frames in real_frames:
            edges.append(torch.combinations(frames, with_replacement=True).T)


        pc_embeddings = None
        if self.backbone is not None and self.pc is not None:
            if metadata_pc==None or pc_edge_index == None:
                raise ValueError("Point cloud and edge index are required for the pointcloud model")    
            video_embedding = self.backbone(vid)
            # create a fully connected point cloud 
            metadata_pc_obj = Data(x=pc_features, pos=metadata_pc, edge_index=pc_edge_index)
            pc_embeddings = self.pc(metadata_pc_obj)
            node_features = torch.cat((video_embedding * mask.unsqueeze(-1).float(),
                                       pc_embeddings.unsqueeze(1).unsqueeze(1).expand(video_embedding.shape[0], 
                                                                                      video_embedding.shape[1], 
                                                                                      video_embedding.shape[2], 
                                                                                      -1)), 
                                                                                      dim=-1)
        
        elif self.backbone is not None:
            video_embedding = self.backbone(vid)
            node_features = video_embedding * mask.unsqueeze(-1).float()
    
        # Concatenate the video embedding and the point cloud embedding
        # elif self.pc is not None:
        #     if metadata_pc==None or pc_edge_index == None:
        #         raise ValueError("Point cloud and edge index are required for the pointcloud model")    
        #     metadata_pc_obj = Data(x=pc_features, pos=metadata_pc, edge_index=pc_edge_index)
        #     pc_embeddings = self.pc(metadata_pc_obj)          
        #     # Really hacky way of doing this. We need to fix this based on the mask 
        #     # Here we are assuming that the number of frames is the same as the number of nodes in the graph
        #     num_frames = (gnn_edge_index[ 0, -1, -1] + 1).item()
        #     node_features = torch.cat((video_embedding,
        #                                # This is a really hacky way of doing this.
        #                                pc_embeddings.unsqueeze(1).expand(-1, 
        #                                                                  num_frames, 
        #                                                                  -1)), 
        #                                                                  dim=-1)
        else:
            raise NotImplementedError("At least one of the submodels pc or backbone must be initialized")
            
        node_features = node_features.view(node_features.shape[0],-1, node_features.shape[-1])

        data_list = [Data(x=node_features[i], edge_index=edges[i]) for i in range(batch_size)] 
        batch = Batch.from_data_list(data_list)
        # pass the graph through the GNN
        # TODO: Make sure the softmax is applied to the right dim and check if we need logits here
        y_pred = self.gnn(batch)
        return y_pred
        
            
    def get_sub_models(self):
        return self.sub_models
    
    def get_model_params(self):
        param_dict = dict()
        for sub_model in self.get_sub_models():
            param_dict.update({"params": getattr(self, sub_model).parameters()})
        return param_dict

class ResNet3DBackbone(nn.Module):
    def __init__(self, pretrained, num_frames, num_vids=2, embedding_size=128):
        super(ResNet3DBackbone, self).__init__()
        self.embed_dim = embedding_size
        self.model = models.r2plus1d_18(pretrained=pretrained)
        # Modify the last layer of the model to output embeddings
        self.fc = nn.Sequential(
            nn.Linear(in_features=400, out_features=num_vids*num_frames*self.embed_dim //4 , bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=num_vids*num_frames*self.embed_dim //4, out_features=num_vids*num_frames*self.embed_dim //2 , bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=num_vids*num_frames*self.embed_dim //2, out_features=num_vids*num_frames*self.embed_dim, bias=True))

        # Freeze all the layers except the last layer
        for name, param in self.model.named_parameters():
            if name != 'fc.weight' and name != 'fc.bias':
                param.requires_grad = False

    # Define a function to extract embeddings from an input video
    def forward(self, video):
        # Set themodel to evaluation mode
        self.model.eval()

        batch_size, num_vids, num_frames, height, width, channels = video.shape
        
        video = video.contiguous().view(
            video.shape[0], video.shape[1] * video.shape[2], video.shape[3], video.shape[4], video.shape[5]
        )
        video = video.permute(0, 4, 1, 2, 3)

        # Pass the video through the model
        with torch.no_grad():
            embeddings = self.fc(self.model(video))
        
        embeddings = embeddings.reshape(batch_size, num_vids, num_frames, self.embed_dim)
        # Return the embeddings
        return embeddings

class GNNClassifier(nn.Module):
    def __init__(self, channel_list, num_classes, heads, do_prob=0.5):
        super(GNNClassifier, self).__init__()
        self.p = do_prob

        self.gnn_layers = nn.ModuleList()
        # Add the input layer
        self.gnn_layers.append(GATConv(channel_list[0], channel_list[1], heads[0]))
        for i in range(1, len(channel_list) - 1):
            self.gnn_layers.append(GATConv(in_channels=channel_list[i]*heads[i-1],
                                           out_channels=channel_list[i+1], 
                                           heads=heads[i],
                                           dropout=do_prob))
        self.softmax = nn.Softmax(dim=-1)
            
        self.fc = nn.Linear(channel_list[-1]*heads[-1], num_classes)

    def forward(self, data):
        # x: (batch_size, embedding_size)
        x, edge_index = data.x, data.edge_index
        # Compute GNN features
        for gnn_layer in self.gnn_layers[:-1]:
            x = gnn_layer(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.p, training=self.training)

        # apply the last layer
        x = self.gnn_layers[-1](x, data.edge_index)

        # Reshape the data back to the oiiginal shape
        x = x.reshape(data.num_graphs, -1, x.shape[-1])
        # Compute final output
        # y = self.softmax(self.fc(x.mean(dim=1)))
        # Global average pooling over nodes
        x = self.fc(x.mean(dim=1))
        # y = self.softmax(x)
        return x
     
class PointCloud(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_layers,
                 hidden_dim,
                 num_heads,
                 dropout):
        super(PointCloud, self).__init__()

        # Use MLP to compute positional encoding
        self.pos_nn = MLP([in_channels, hidden_dim, hidden_dim, out_channels])
        # self.attn_nn = MLP([in_channels, hidden_dim, hidden_dim, out_channels])

        self.point_transformer = PointTransformerConv(in_channels=in_channels, 
                                                      out_channels=out_channels, 
                                                      num_layers=num_layers, 
                                                      pos_nn=self.pos_nn,
                                                      num_heads=num_heads, 
                                                      add_self_loops=False,
                                                      dropout=dropout)    
    def forward(self, x):
        # x: (batch_size, num_frames, 1, height, width)
        # edge_index: (2, num_edges)
        embeddings = []
        for i in range(x.x.shape[0]):
            embeddings.append(self.point_transformer(x=x.x[i].float(), 
                                                    pos=x.pos[i].float(), 
                                                    edge_index=x.edge_index[i]).unsqueeze(0))
        
        # Concatenate all the embeddings
        return torch.cat(embeddings).mean(dim=1) # (Batch size x out_channels)

