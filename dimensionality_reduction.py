import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap
from sklearn.cluster import KMeans
from argparse import Namespace

class DimensionReductionAndClustering:
    def __init__(self, args):
        self.embedding_path = args.dataset
        self.trained_Emb = np.load(self.embedding_path)

        if args.reduce_dimension:
            if args.dimensionality_reduction_type == 'pca':
                self.reduced_dim = PCA(n_components=args.reduced_dimension_components)
            elif args.dimensionality_reduction_type == 'tsne':
                self.reduced_dim = TSNE(n_components=args.reduced_dimension_components, learning_rate='auto', init = 'random')
            elif args.dimensionality_reduction_type == 'spectral_embedding':
                self.reduced_dim = SpectralEmbedding(n_components=args.reduced_dimension_components)
            elif args.dimensionality_reduction_type == 'ISOMAP':
                self.reduced_dim = Isomap(n_components=args.reduced_dimension_components)
            else:
                self.reduced_dim = PCA(args.reduced_dimension_components)
        else:
            #default choose pca
            self.reduced_dim = PCA(n_components=args.reduced_dimension_components)

        self.trained_Emb = self.reduced_dim.fit_transform(self.trained_Emb)

        self.kmeans = KMeans(n_clusters=args.number_of_clusters,init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=1234)
        self.kmeans.fit(self.trained_Emb)

    def save_embeddings(self, save_path):
        np.save(save_path, self.trained_Emb)

if __name__ == '__main__':
    args = Namespace(
        dataset='dataset/umls/entity_embedding_st.npy',
        reduce_dimension=True,
        dimensionality_reduction_type='pca',
        reduced_dimension_components=2,
        number_of_clusters=3,
    )
    model = DimensionReductionAndClustering(args)
    reduced_embedding = model.trained_Emb
    print(f"the reduced embedding shape is {reduced_embedding.shape}")
    cluster_object = model.kmeans
    model.save_embeddings(save_path='dataset/umls/entity_embedding_dimensionality_reduced.npy')



