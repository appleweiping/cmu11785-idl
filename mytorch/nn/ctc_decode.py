"""CTC decoders for MyTorch (CMU 11-785 HW3P1): greedy and beam search.

Both take softmax probabilities of shape ``(seq_len, batch, num_symbols)`` and a
``symbol_set`` (the non-blank labels; blank is index 0).
"""

import numpy as np


class GreedySearchDecoder:
    def __init__(self, symbol_set):
        self.symbol_set = symbol_set          # e.g. ["a", "b", "c"]

    def decode(self, y_probs):
        """
        y_probs: (num_symbols, seq_len, batch)   [HW3P1 layout]
        Returns (decoded_string, path_prob) for batch index 0.
        """
        num_sym, T, B = y_probs.shape
        best = y_probs[:, :, 0]                # (num_symbols, T)
        path_prob = 1.0
        idx_path = []
        for t in range(T):
            k = int(np.argmax(best[:, t]))
            path_prob *= best[k, t]
            idx_path.append(k)

        # collapse repeats then remove blanks (blank == index 0)
        decoded = []
        prev = None
        for k in idx_path:
            if k != prev:
                if k != 0:
                    decoded.append(self.symbol_set[k - 1])
                prev = k
        return "".join(decoded), path_prob


class BeamSearchDecoder:
    def __init__(self, symbol_set, beam_width):
        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        y_probs: (num_symbols, seq_len, batch)
        Returns (best_path, merged_path_scores) for batch index 0.
        """
        num_sym, T, B = y_probs.shape
        y = y_probs[:, :, 0]
        blank = 0

        # beams keyed by prefix string; track prob ending in blank / in symbol
        # PathScore for paths ending in a symbol; BlankPathScore for ending blank
        paths_blank = {"": y[blank, 0]}
        paths_sym = {}
        for c in range(1, num_sym):
            paths_sym[self.symbol_set[c - 1]] = y[c, 0]

        def prune(pb, ps):
            scores = list(pb.values()) + list(ps.values())
            if not scores:
                return pb, ps
            scores.sort(reverse=True)
            cutoff = scores[min(self.beam_width, len(scores)) - 1]
            npb = {p: v for p, v in pb.items() if v >= cutoff}
            nps = {p: v for p, v in ps.items() if v >= cutoff}
            return npb, nps

        paths_blank, paths_sym = prune(paths_blank, paths_sym)

        for t in range(1, T):
            new_blank, new_sym = {}, {}
            # extend with blank
            for p, v in paths_blank.items():
                new_blank[p] = new_blank.get(p, 0.0) + v * y[blank, t]
            for p, v in paths_sym.items():
                new_blank[p] = new_blank.get(p, 0.0) + v * y[blank, t]
            # extend with symbols
            for c in range(1, num_sym):
                sym = self.symbol_set[c - 1]
                # from blank-ending paths -> always append
                for p, v in paths_blank.items():
                    np_ = p + sym
                    new_sym[np_] = new_sym.get(np_, 0.0) + v * y[c, t]
                # from symbol-ending paths
                for p, v in paths_sym.items():
                    np_ = p if (p and p[-1] == sym) else p + sym
                    new_sym[np_] = new_sym.get(np_, 0.0) + v * y[c, t]

            paths_blank, paths_sym = prune(new_blank, new_sym)

        # merge identical prefixes
        merged = dict(paths_blank)
        for p, v in paths_sym.items():
            merged[p] = merged.get(p, 0.0) + v
        best_path = max(merged, key=merged.get)
        return best_path, merged
