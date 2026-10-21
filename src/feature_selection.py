from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Knihovni nastroje jsou zde pouzity zamerne a v omezene mire: k-means, kNN
# a silueta jsou periferie (siluetu jsme implementovali v cv3), tezistem cv5
# je logika vyberu priznaku, ne re-implementace techto algoritmu.
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class FeatureSelector(ABC):
    """Spolecne rozhrani vsech selektoru priznaku.

    Az dosud byl cely kurz bez ucitele (unsupervised) — pracoval pouze s matici
    priznaku ``X``. Zde poprve vstupuje do hry cilova promenna ``y``
    (diagnoza: 0 = benigni, 1 = maligni). Filtrovaci selektor ji pouziva
    ke srovnani obou diagnostickych skupin statistickym testem, wrapper ji
    pouziva jako cil optimalizace. Je to mekky prechod k uceni s ucitelem
    (supervised) — jeste bez klasifikace samotne.
    """

    @abstractmethod
    def select(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vybere priznaky a vrati pole jejich indexu.

        Parameters
        ----------
        x:
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y:
            Binarni cilovy vektor delky ``n_samples`` (hodnoty 0 a 1).

        Returns
        -------
        np.ndarray
            1D pole celociselnych indexu vybranych sloupcu ``x``
            (poradi odpovida sloupcum ``x``).
        """


class FilterSelector(FeatureSelector):
    """Filtrovaci vyber priznaku pomoci dvouvyberovych statistickych testu.

    Selektor hodnoti kazdy priznak nezavisle na ostatnich a ponechava jen ty,
    ktere se mezi skupinami ``y == 0`` a ``y == 1`` lisi ZAROVEN
    *vyznamne* (mala p-hodnota) a *podstatne* (dostatecna velikost ucinku).
    """

    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.5) -> None:
        """Inicializuje selektor.

        Parameters
        ----------
        alpha:
            Hladina vyznamnosti — priznak je "vyznamny", pokud p-hodnota
            testu je mensi nez ``alpha``.
        min_effect_size:
            Minimalni pozadovana absolutni hodnota velikosti ucinku
            (Cohenovo ``d``), aby byl priznak povazovan za "podstatny".
        """
        self.alpha = alpha
        self.min_effect_size = min_effect_size

    def select(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vybere priznaky, ktere obe skupiny oddeluji vyznamne i podstatne."""
        assert isinstance(x, np.ndarray), "Vstup x musí být typu np.ndarray."
        assert x.ndim == 2, "Matice x musí mít 2 rozměry."
        assert len(y) == x.shape[0], "Délka y musí odpovídat počtu řádků x."
        assert set(np.unique(y)).issubset({0, 1}), "Cílový vektor y musí být binární ({0, 1})."

        selected_indices = []
        n_features = x.shape[1]

        for j in range(n_features):
            a = x[y == 0, j]
            b = x[y == 1, j]

            # 1. Test normality obou skupin
            res_a = stats.shapiro(a)
            res_b = stats.shapiro(b)

            # 2. Volba testu
            if res_a.pvalue > self.alpha and res_b.pvalue > self.alpha:
                p_val = stats.ttest_ind(a, b).pvalue
            else:
                p_val = stats.mannwhitneyu(a, b).pvalue

            # 3. Velikost účinku
            d = self._cohens_d(a, b)

            # 4. Ponechání podle obou kritérií
            if p_val < self.alpha and abs(d) >= self.min_effect_size:
                selected_indices.append(j)

        return np.array(selected_indices, dtype=int)

    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Vrati Cohenovo ``d`` — velikost ucinku pro dva nezavisle vybery."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n1 = a.size
        n2 = b.size
        pooled_var = (
            (n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)
        ) / (n1 + n2 - 2)
        pooled_std = float(np.sqrt(pooled_var))
        return float((a.mean() - b.mean()) / pooled_std)


class WrapperSelector(FeatureSelector, ABC):
    """Obalovy (wrapper) vyber priznaku hladovou doprednou selekci."""

    def __init__(self, target: float) -> None:
        self.target = target

    def select(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Hladove pridava priznaky, dokud skore nedosahne ``self.target``."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        n_features = x.shape[1]
        for i in range(1, n_features + 1):
            score = self._score(x[:, :i], y)
            if score >= self.target:
                return np.arange(i)
        return np.arange(n_features)

    @abstractmethod
    def _score(self, x_subset: np.ndarray, y: np.ndarray) -> float:
        """Vrati skore podmnoziny priznaku ``X_subset`` (vetsi = lepsi)."""


class SilhouetteWrapperSelector(WrapperSelector):
    """Wrapper skorujici podmnozinu priznaku siluetou k-means shlukovani."""

    def __init__(
        self,
        target: float,
        n_clusters: int = 2,
        random_state: int = 0,
    ) -> None:
        super().__init__(target)
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _score(self, x_subset: np.ndarray, y: np.ndarray) -> float:
        """Vrati siluetu k-means shlukovani na ``x_subset``."""
        assert isinstance(x_subset, np.ndarray), "x_subset musí být typu np.ndarray."
        assert x_subset.ndim == 2, "x_subset musí mít 2 rozměry."
        assert x_subset.shape[0] > self.n_clusters, "Počet vzorků musí být větší než n_clusters."

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(x_subset)
        return float(silhouette_score(x_subset, labels))


class KNNWrapperSelector(WrapperSelector):
    """Wrapper skorujici podmnozinu priznaku testovaci presnosti kNN."""

    def __init__(
        self,
        target: float,
        n_neighbors: int = 3,
        test_size: float = 0.2,
        random_state: int = 0,
    ) -> None:
        super().__init__(target)
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state

    def _score(self, x_subset: np.ndarray, y: np.ndarray) -> float:
        """Vrati testovaci presnost kNN klasifikatoru na ``x_subset``."""
        assert isinstance(x_subset, np.ndarray), "x_subset musí být typu np.ndarray."
        assert x_subset.ndim == 2, "x_subset musí mít 2 rozměry."
        assert len(y) == x_subset.shape[0], "Délka y musí odpovídat počtu vzorků v x_subset."
        assert len(np.unique(y)) >= 2, "Vektor y musí obsahovat alespoň 2 třídy."

        x_tr, x_te, y_tr, y_te = train_test_split(
            x_subset,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        clf.fit(x_tr, y_tr)
        return float(clf.score(x_te, y_te))