class CorrelationTransformer(TransformerMixin, BaseEstimator):
    """ A transformer that removes correlated features.
    
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, thx_excl = .95):
        self.thx_excl = thx_excl

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=False)
        
        self.n_features_ = X.shape[1]
        self.n_features_in_ = X.shape[1]
        
        self.X_corr_ = self.corr_mat_lower(X)
        self.features_drop_idx_, self.features_keep_idx_ = self.drop_correlated_features(X)
        
        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_constrained_ : array, shape (n_samples, n_features)
            The array only containing the to be kept features in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True)
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        X_constrained_ = np.delete(X, self.features_drop_idx_, axis = 1)
            
        return X_constrained_
    
    def fit_transform(self, X):
        """ Fit and transform the CorrelationTransformer
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed_ : array, shape (n_samples, n_features - len(self.feats_drop_idx_))
            The array only containing the to be kept features in ``X``.
        """
        self.fit(X)
        X_constrained_ = self.transform(X)
        return X_constrained_
    
    def corr_mat_lower(self, df):
        """Correlation matrix of a pd.DataFrame with values in lower triangle set to 0.
        Parameters
        ----------
        df : np.array, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        corr_mat : pd.DataFrame, shape (n_features, n_features)
            The pd.DataFrame returning the pairwise Pearson correlations between columns in ``df``.
        """
        self.assert_df_notna(df)
        
        arr_corr = np.corrcoef(df, rowvar = False)
        array_mask = np.array(arr_corr, dtype = bool)
        array_triu = np.triu(array_mask, k = 0)
        array_corr[array_triu] = 0
        return pd.DataFrame(array_corr)
    
    def assert_df_notna(self, df):
        assert np.isnan(df).sum().sum() == 0, "nas cannot be handled"
        
    def drop_correlated_features(self, df):
        """Drop features correlated above thx from df
        Parameters
        ----------
        df : np.array, shape (n_samples, n_features). The input samples.
        df_corr : pd.DataFrame, shape (n_features, n_features). The lower triangle of the correlation matrix of ``df``
        thx_excl : float, threshold above which one of the correlated features is excluded
        Returns
        -------
        df_constrained : pd.DataFrame, shape (n_samples, n_features - len(feats_drop))
            The pd.DataFrame returning the constrained ``df``.
        """
        # input check
        iterable_test = map(self.assert_df_notna, [df, self.X_corr_])
        _ = list(iterable_test)
        assert type(self.thx_excl) == float, "thx_excl has to be dtype float"

        df_corr_long = (
            self.X_corr_.melt(ignore_index = False, var_name = "col_name").
            reset_index().
            rename(columns = {"index":"row_name"})
        )
        df_above_thx = df_corr_long[df_corr_long["value"] > self.thx_excl].sort_values("value", ascending = False)
        feats_drop_idx = np.unique(df_above_thx[["row_name", "col_name"]].max(axis = 1).reset_index(drop = True))
        feats_keep_idx = [c for c in range(0, self.X_corr_.shape[1]) if c not in feats_drop_idx]
        return feats_drop_idx, feats_keep_idx
