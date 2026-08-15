# compositional_lan_tokenizer.py
from typing import List, Dict, Optional
import io
import chess, chess.pgn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from .base_tokenizer import BaseTokenizer

_RESULT = {"1-0","0-1","1/2-1/2","*"}
FILES = "abcdefgh"
RANKS = "12345678"
SQUARES = [f+r for f in FILES for r in RANKS]
PROMOS  = "QRBN"
DIGITS = set("0123456789")

def _vocab(include_move_numbers: bool, keep_result: bool,
           bos: str, eos: str, unk: str, pad: str = None) -> Dict[str,int]:
    base = [bos, eos, unk]
    if pad is not None:
        base.append(pad)
    ops  = ["x","=","+","#","O-O","O-O-O",".","..."]
    toks = base + list("KQRBNP") + SQUARES + list(PROMOS) + ops
    if include_move_numbers: toks += list("0123456789")
    if keep_result: toks += sorted(_RESULT)
    return {t:i for i,t in enumerate(dict.fromkeys(toks))}

class LanTokenizer(BaseTokenizer):
    def __init__(self, config: Optional[dict] = None):
        include_move_numbers = config.get("include_move_numbers", True)
        include_black_tripledots = config.get("include_black_tripledots", False)
        bos = config.get("bos", "<bos>")
        eos = config.get("eos", "<eos>")
        unk = config.get("unk", "<unk>")
        pad = config.get("pad", None)
        keep_result = config.get("keep_result", False)
        self._bos, self._eos, self._unk, self._pad = bos, eos, unk, pad
        self._keep_res = keep_result
        self._include_nums = include_move_numbers
        self._include_black_ellipses = include_black_tripledots

        tok2id = _vocab(include_move_numbers, keep_result, bos, eos, unk, pad)
        self._tok2id = tok2id
        self.tk = Tokenizer(WordLevel(vocab=tok2id, unk_token=self._unk))
        self.tk.pre_tokenizer = WhitespaceSplit()

    def _pgn_to_tokens(self, text: str) -> List[str]:
        import os, contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            g = chess.pgn.read_game(io.StringIO(text))
        if g is None:
            return text.split()

        b, out, n = g.board(), [], 1
        for mv in g.mainline_moves():
            if b.turn == chess.WHITE and self._include_nums:
                out += list(str(n)) + (["..."] if self._include_black_ellipses and b.fullmove_number<n else ["."])

            if b.is_castling(mv):
                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                b.pop()
                out.append("O-O" if chess.square_file(mv.to_square) == 6 else "O-O-O")
                if suf: out.append(suf)
                b.push(mv)
            else:
                piece = b.piece_at(mv.from_square).symbol().upper()
                frm = chess.square_name(mv.from_square)
                to  = chess.square_name(mv.to_square)
                is_cap = b.is_capture(mv)
                promo = mv.promotion

                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                # emit LAN tokens
                out.append(piece)
                out.append(frm)
                if is_cap: out.append("x")
                out.append(to)
                if promo: out += ["=", chess.piece_symbol(promo).upper()]
                if suf: out.append(suf)
            if b.turn == chess.WHITE: n += 1

        res = g.headers.get("Result")
        if self._keep_res and res in _RESULT: out.append(res)
        return out

    def encode(self, text: str) -> List[int]:
        toks = [self._bos] + self._pgn_to_tokens(text) + [self._eos]
        return self.tk.encode(" ".join(toks)).ids

    def decode(self, ids: List[int]) -> str:
        toks = [t for t in self.tk.decode(ids).split() if t not in {self._bos, self._eos}]
        out: List[str] = []
        i, n = 0, len(toks)

        while i < n:
            t = toks[i]

            if t and all(ch in DIGITS for ch in t):
                j = i
                num = []
                while j < n and all(ch in DIGITS for ch in toks[j]):
                    num.append(toks[j]); j += 1
                dots = ""
                if j < n and toks[j] in {".", "..."}:
                    dots = toks[j]; j += 1
                out.append("".join(num) + dots)
                i = j
                continue

            if t in {"O-O","O-O-O"}:
                j = i + 1
                suf = toks[j] if j < n and toks[j] in {"+","#"} else ""
                if suf: j += 1
                out.append(t + suf)
                i = j
                continue

            if t in set("KQRBNP"):
                piece = t
                j = i + 1
                frm = toks[j] if j < n else ""; j += 1
                cap = ""
                if j < n and toks[j] == "x":
                    cap = "x"; j += 1
                to = toks[j] if j < n else ""; j += 1
                promo = ""
                if j + 1 <= n - 1 and toks[j] == "=" and toks[j+1] in set(PROMOS):
                    promo = "=" + toks[j+1]; j += 2
                suf = ""
                if j < n and toks[j] in {"+","#"}:
                    suf = toks[j]; j += 1
                lan = f"{piece}{frm}{cap}{to}{promo}{suf}"
                out.append(lan)
                i = j
                continue

            out.append(t); i += 1

        return " ".join(out)

    def get_vocab(self) -> Dict[str,int]: return self._tok2id
    def bos_id(self) -> Optional[int]: return self._tok2id[self._bos]
    def eos_id(self) -> Optional[int]: return self._tok2id[self._eos]
    def pad_id(self) -> Optional[int]: return self._tok2id.get(self._pad) if self._pad else None
    def get_vocab_size(self) -> int: return len(self._tok2id)