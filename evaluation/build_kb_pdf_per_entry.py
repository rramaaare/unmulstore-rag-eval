"""Buat PDF knowledge base dengan SATU entri Q&A per halaman.

Tujuannya: tiap entri menjadi satu potongan (chunk) sendiri di vector store, tidak lagi bercampur dengan
entri tetangga seperti pada PDF lama (potongan ±1000 karakter yang memuat median 3 entri).

Format teks tiap entri sama dengan knowledge base asli (judul, Varian, Jawaban, Domain | Tags).
"""
import argparse
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from build_testset import parse, read_paragraphs
import re


def read_domain_tags(path):
    """Ambil baris 'Domain: ... | Tags: ...' asli per nomor Q (parse() hanya menyimpan domain pertama)."""
    paras = read_paragraphs(path)
    out = {}
    for i, p in enumerate(paras):
        m = re.match(r"Q(\d+)\. ", p)
        if m and i + 3 < len(paras) and paras[i + 2].startswith("Jawaban:"):
            out[int(m.group(1))] = re.sub(r"\s+\|\s+", " | ", paras[i + 3].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", default="../FAQ_UnmulStore_KnowledgeBase_updated.docx")
    ap.add_argument("--out", default="FAQ_per_entry.pdf")
    ap.add_argument("--pages-per-file", type=int, default=0, help="0 = satu PDF; >0 = pecah per N entri")
    a = ap.parse_args()

    entries = parse(a.docx)
    tags = read_domain_tags(a.docx)

    if a.pages_per_file:
        # pecah menjadi beberapa berkas kecil (mis. untuk menghindari batas kecepatan embedding saat unggah)
        base = a.out.rsplit(".", 1)[0]
        n_files = 0
        for k in range(0, len(entries), a.pages_per_file):
            chunk = entries[k: k + a.pages_per_file]
            n_files += 1
            name = f"{base}_bagian{n_files}.pdf"
            cv = canvas.Canvas(name, pagesize=A4)
            cv.setTitle(f"FAQ UnmulStore bagian {n_files}")
            for e in chunk:
                block = [f"Q{e['id']}. {e['title']}", "Varian: " + " | ".join(e["varian"]),
                         "Jawaban: " + e["answer"], tags.get(e["id"], f"Domain: {e['domain']}")]
                block = [line.replace("→", "->") for line in block]
                y = A4[1] - 2 * cm
                for line in block:
                    for part in textwrap.wrap(line, 95) or [""]:
                        cv.drawString(2 * cm, y, part)
                        y -= 0.6 * cm
                cv.showPage()
            cv.save()
            print(f"{name}: {len(chunk)} entri (Q{chunk[0]['id']} - Q{chunk[-1]['id']})")
        return

    c = canvas.Canvas(a.out, pagesize=A4)
    c.setTitle("FAQ UnmulStore - satu entri per halaman")
    lengths = []
    for e in entries:
        block = [
            f"Q{e['id']}. {e['title']}",
            "Varian: " + " | ".join(e["varian"]),
            "Jawaban: " + e["answer"],
            tags.get(e["id"], f"Domain: {e['domain']}"),
        ]
        # font standar PDF (WinAnsi) tidak punya tanda panah; ganti agar teks tidak hilang/menjadi kotak
        block = [line.replace("→", "->") for line in block]
        text = "\n".join(block)
        lengths.append(len(text))
        y = A4[1] - 2 * cm
        for line in block:
            for part in textwrap.wrap(line, 95) or [""]:
                c.drawString(2 * cm, y, part)
                y -= 0.6 * cm
        c.showPage()
    c.save()
    lengths.sort()
    print(f"{len(entries)} entri -> {a.out} ({len(entries)} halaman)")
    print(f"panjang teks per entri: median {lengths[len(lengths)//2]}, maks {lengths[-1]} karakter, "
          f"entri > 900 karakter: {sum(l > 900 for l in lengths)}")


if __name__ == "__main__":
    main()
