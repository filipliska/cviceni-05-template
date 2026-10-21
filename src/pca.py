from __future__ import annotations

import numpy as np


class PCA:
    """Induktivni PCA ve stylu sklearn (nededi z niceho).

    ``fit`` z trenovacich dat nauci prumer, vlastni cisla a vlastni vektory
    (hlavni komponenty) kovariancni matice; ``transform`` timto naucenym
    pravidlem promita libovolna nova data. To je zamerny kontrast s
    transduktivnim shlukovanim z cv4, ktere zadne prenositelne pravidlo nema.
    """

    def __init__(
        self,
        variance_threshold: float | None = 95.0,
        n_components: int | None = None,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None

    def _covariance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Spocita kovariancni matici priznaku ve vektorizovane podobe."""
        assert isinstance(x, np.ndarray), "Vstup x musí být typu np.ndarray."
        assert x.ndim == 2, "Vstup x musí mít 2 rozměry (vzorky x příznaky)."
        m = x.shape[0]
        assert m >= 2, "Vstup x musí obsahovat alespoň 2 vzorky (m >= 2)."

        xc = x - x.mean(axis=0)
        return (xc.T @ xc) / (m - 1)

    def _select_components(self, eigenvalues: np.ndarray) -> int:
        """Urci pocet komponent podle prahu kumulativniho rozptylu."""
        assert isinstance(eigenvalues, np.ndarray), "eigenvalues musí být np.ndarray."
        assert eigenvalues.ndim == 1, "eigenvalues musí být 1D pole."
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:]), "eigenvalues musí být seřazena sestupně."
        assert self.variance_threshold is not None and 0 < self.variance_threshold <= 100, "variance_threshold musí být v rozmezí (0, 100]."

        ratio = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(ratio) * 100
        
        idx = np.where(cumulative >= self.variance_threshold)[0]
        if len(idx) > 0:
            return int(idx[0] + 1)
        return len(eigenvalues)

    def fit(self, x: np.ndarray) -> "PCA":
        """Nauci PCA model z trenovacich dat ``x``."""
        self.mean_ = x.mean(axis=0)

        cov = self._covariance_matrix(x)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.eigenvalues_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()

        if self.n_components is not None:
            self.n_components_ = self.n_components
        else:
            self.n_components_ = self._select_components(self.eigenvalues_)

        self.components_ = eigenvectors[:, : self.n_components_]
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Nauci model a rovnou vrati promitnuta data."""
        return self.fit(x).transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Promitne data ``x`` do prostoru naucenych hlavnich komponent."""
        assert self.components_ is not None, "Model ještě nebyl naučen (fit nebyl spuštěn)."
        assert isinstance(x, np.ndarray), "Vstup x musí být typu np.ndarray."
        assert x.ndim == 2, "Vstup x musí mít 2 rozměry."
        assert x.shape[1] == self.mean_.shape[0], "Počet příznaků v x neodpovídá trénovacím datům."

        return (x - self.mean_) @ self.components_

    def inverse_transform(self, x_pca: np.ndarray) -> np.ndarray:
        """Rekonstruuje data z prostoru hlavnich komponent zpet do puvodniho."""
        assert self.components_ is not None, "Model ještě nebyl naučen."
        assert isinstance(x_pca, np.ndarray), "Vstup x_pca musí být typu np.ndarray."
        assert x_pca.ndim == 2, "Vstup x_pca musí mít 2 rozměry."
        assert x_pca.shape[1] == self.n_components_, "Počet komponent v x_pca neodpovídá n_components_ modelu."

        return x_pca @ self.components_.T + self.mean_

    def save(self, path: str) -> None:
        """Ulozi naucený model do souboru ``.npz`` pomoci ``np.savez``."""
        assert self.components_ is not None, "Nelze uložit nenaučený model."
        np.savez(
            path,
            components_=self.components_,
            mean_=self.mean_,
            eigenvalues_=self.eigenvalues_,
            explained_variance_ratio_=self.explained_variance_ratio_,
            n_components_=self.n_components_,
        )

    @classmethod
    def load(cls, path: str) -> "PCA":
        """Nacte model ulozený metodou ``save`` a vrati hotovou instanci."""
        data = np.load(path)
        assert "components_" in data and "mean_" in data, "Soubor neobsahuje povinná pole components_ nebo mean_."
        
        components = data["components_"]
        assert components.ndim == 2, "Načtené komponenty musí mít 2 rozměry."

        obj = cls()
        obj.components_ = components
        obj.mean_ = data["mean_"]
        if "eigenvalues_" in data:
            obj.eigenvalues_ = data["eigenvalues_"]
        if "explained_variance_ratio_" in data:
            obj.explained_variance_ratio_ = data["explained_variance_ratio_"]
        if "n_components_" in data:
            obj.n_components_ = int(data["n_components_"])
            
        return obj