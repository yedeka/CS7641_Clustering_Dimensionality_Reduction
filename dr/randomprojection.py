from sklearn import random_projection


def apply_rp(features, components,indicator):
    print('Random projection for ',indicator)
    transformer = random_projection.GaussianRandomProjection(n_components=components, random_state=150)
    reduced_features = transformer.fit_transform(features)
    print(reduced_features)