from .degree_fair import DFairGNN_2, DFair_GAT, DFair_GCN, DFair_Sage
from .base_gnn import GCNWithIntermediateOutput, GATWithIntermediateOutput, GraphSAGEWithIntermediateOutput, MLP, RandomModel
from .degree_discriminator import RandomDegreeDiscriminator
from .linear_gnn import LinearGNN
from .all_gnn import AllGNN, LinkPredictor
from .discriminator_tailgnn import DiscriminatorTailGNN