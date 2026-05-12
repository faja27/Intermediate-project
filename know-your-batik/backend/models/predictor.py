import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Make src/ importable from the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import get_model
from src.preprocessor import get_transforms

BATIK_INFO: dict[str, dict] = {
    "Bali_Barong": {
        "origin": "Bali",
        "region": "Bali",
        "description": "Batik Bali Barong menampilkan motif Barong, makhluk mitologi Bali yang melambangkan kebaikan dan pelindung alam semesta. Motif ini kaya akan ornamen tradisional Bali dengan warna-warna cerah.",
        "characteristics": [
            "Motif Barong sebagai figur utama",
            "Warna cerah dan kontras seperti merah, emas, dan hitam",
            "Detail ornamen ukiran Bali yang rumit",
            "Mengandung simbol spiritual Hindu Bali",
            "Perpaduan motif flora dan fauna khas Bali",
        ],
    },
    "Bali_Merak": {
        "origin": "Bali",
        "region": "Bali",
        "description": "Batik Bali Merak menampilkan motif burung merak yang anggun, melambangkan keindahan, keanggunan, dan kemakmuran dalam budaya Bali.",
        "characteristics": [
            "Motif burung merak dengan ekor yang mengembang",
            "Warna-warna cerah dan mencolok",
            "Detail bulu merak yang sangat halus",
            "Ornamen bunga dan daun sebagai latar",
            "Perpaduan warna biru, hijau, dan emas",
        ],
    },
    "Ceplok": {
        "origin": "Jawa Tengah",
        "region": "Solo dan Yogyakarta",
        "description": "Batik Ceplok merupakan salah satu motif batik klasik Jawa yang memiliki pola geometris berupa kotak, bintang, atau bentuk simetris yang berulang.",
        "characteristics": [
            "Pola geometris yang simetris dan berulang",
            "Motif lingkaran, bintang, atau kotak",
            "Warna tradisional: sogan, biru, dan putih",
            "Garis tegas dan presisi",
            "Dipengaruhi motif batik keraton",
        ],
    },
    "Corak_Insang": {
        "origin": "Kalimantan Barat",
        "region": "Pontianak, Kalimantan Barat",
        "description": "Batik Corak Insang berasal dari Kalimantan Barat dengan motif yang terinspirasi dari sisik ikan yang melambangkan kesuburan dan kehidupan masyarakat Melayu Pontianak.",
        "characteristics": [
            "Motif menyerupai sisik ikan atau insang",
            "Warna dominan kuning, merah, dan hitam",
            "Susunan motif yang rapat dan teratur",
            "Pengaruh budaya Melayu Pontianak",
            "Makna kesuburan dan keberkahan",
        ],
    },
    "Ikat_Celup": {
        "origin": "Berbagai daerah Indonesia",
        "region": "Nasional",
        "description": "Batik Ikat Celup dibuat dengan teknik mengikat dan mencelupkan kain ke pewarna, menghasilkan pola abstrak yang unik dan tidak berulang persis sama.",
        "characteristics": [
            "Teknik pembuatan dengan ikat dan celup",
            "Pola abstrak dan organik",
            "Setiap kain menghasilkan pola unik",
            "Gradasi warna yang natural",
            "Tekstur visual yang dinamis",
        ],
    },
    "Jakarta_Ondel_Ondel": {
        "origin": "DKI Jakarta",
        "region": "Jakarta",
        "description": "Batik Jakarta Ondel-Ondel menampilkan motif boneka raksasa Ondel-Ondel, ikon budaya Betawi yang berfungsi sebagai penolak bala dalam tradisi masyarakat Jakarta.",
        "characteristics": [
            "Motif Ondel-Ondel sebagai ikon Betawi",
            "Warna cerah merah dan putih dominan",
            "Ornamen khas budaya Betawi",
            "Latar belakang motif batik tradisional",
            "Melambangkan penjaga dan pelindung",
        ],
    },
    "Jawa_Barat_Megamendung": {
        "origin": "Jawa Barat",
        "region": "Cirebon, Jawa Barat",
        "description": "Batik Megamendung adalah batik khas Cirebon yang menampilkan motif awan berlapis-lapis. Motif ini merupakan warisan budaya yang diakui UNESCO dan dipengaruhi budaya Tiongkok.",
        "characteristics": [
            "Motif awan berlapis-lapis (mega mendung)",
            "Gradasi warna dari gelap ke terang",
            "Pengaruh budaya Tiongkok",
            "Warna biru, merah, atau coklat khas",
            "Warisan budaya tak benda UNESCO",
        ],
    },
    "Jawa_Timur_Pring": {
        "origin": "Jawa Timur",
        "region": "Situbondo, Jawa Timur",
        "description": "Batik Jawa Timur Pring menampilkan motif pohon bambu (pring) yang melambangkan keteguhan, kelenturan, dan keuletan masyarakat Jawa Timur.",
        "characteristics": [
            "Motif dominan pohon bambu (pring)",
            "Warna hijau dan coklat alami",
            "Simbol keteguhan dan keuletan",
            "Latar belakang sederhana dan bersih",
            "Garis-garis vertikal yang elegan",
        ],
    },
    "Kalimantan_Dayak": {
        "origin": "Kalimantan",
        "region": "Kalimantan",
        "description": "Batik Kalimantan Dayak terinspirasi dari motif-motif ukiran dan seni tradisi suku Dayak Kalimantan, kaya dengan simbol-simbol alam dan kepercayaan lokal.",
        "characteristics": [
            "Motif terinspirasi ukiran Dayak",
            "Gambar burung Enggang khas Kalimantan",
            "Warna-warna tanah dan merah",
            "Simbol-simbol kepercayaan lokal Dayak",
            "Motif flora dan fauna hutan Kalimantan",
        ],
    },
    "Lampung_Gajah": {
        "origin": "Lampung",
        "region": "Lampung",
        "description": "Batik Lampung Gajah menampilkan motif gajah Sumatera yang merupakan simbol kebanggaan dan kekuatan masyarakat Lampung, biasanya dipadukan dengan motif tapis khas Lampung.",
        "characteristics": [
            "Motif gajah Sumatera sebagai elemen utama",
            "Perpaduan dengan motif tapis Lampung",
            "Warna coklat, hitam, dan emas",
            "Simbol kekuatan dan kebesaran",
            "Ornamen geometris khas Lampung",
        ],
    },
    "Lasem": {
        "origin": "Jawa Tengah",
        "region": "Lasem, Rembang, Jawa Tengah",
        "description": "Batik Lasem adalah batik pesisir khas Rembang yang dipengaruhi akulturasi budaya Jawa dan Tiongkok, terkenal dengan warna merah darah (abang getih pithik) yang khas.",
        "characteristics": [
            "Warna merah khas 'abang getih pithik'",
            "Perpaduan motif Jawa dan Tiongkok",
            "Motif naga, phoenix, dan bunga peoni",
            "Dibuat dengan teknik batik tulis halus",
            "Simbol akulturasi budaya pesisir",
        ],
    },
    "Madura_Mataketeran": {
        "origin": "Madura",
        "region": "Madura, Jawa Timur",
        "description": "Batik Madura Mataketeran adalah batik khas Madura dengan motif mata keteram (mata hiu) yang memberikan efek visual berulang seperti sisik atau mata ikan.",
        "characteristics": [
            "Motif mata keteram (mata hiu) berulang",
            "Warna cerah dan berani: merah, kuning, biru",
            "Pola geometris dengan motif bulat",
            "Batik tulis dengan detail halus",
            "Karakter warna yang kuat dan ekspresif",
        ],
    },
    "Maluku_Pala": {
        "origin": "Maluku",
        "region": "Maluku",
        "description": "Batik Maluku Pala terinspirasi dari tanaman pala yang merupakan rempah-rempah ikonik Maluku. Motif ini mencerminkan kekayaan alam dan sejarah perdagangan rempah kepulauan Maluku.",
        "characteristics": [
            "Motif buah dan daun pala",
            "Warna hijau, merah, dan coklat",
            "Mencerminkan kekayaan rempah Maluku",
            "Detail tanaman yang naturalistik",
            "Simbol sejarah perdagangan rempah",
        ],
    },
    "NTB_Lumbung": {
        "origin": "Nusa Tenggara Barat",
        "region": "Lombok, NTB",
        "description": "Batik NTB Lumbung menampilkan motif lumbung padi yang melambangkan kemakmuran dan kesejahteraan masyarakat Sasak di Lombok, Nusa Tenggara Barat.",
        "characteristics": [
            "Motif lumbung padi khas Lombok",
            "Warna coklat, kuning, dan hijau",
            "Simbol kemakmuran dan kesejahteraan",
            "Bentuk arsitektur lumbung tradisional",
            "Ornamen geometris khas NTB",
        ],
    },
    "Papua_Asmat": {
        "origin": "Papua",
        "region": "Asmat, Papua",
        "description": "Batik Papua Asmat terinspirasi dari ukiran kayu suku Asmat Papua yang terkenal di seluruh dunia. Motif ini merepresentasikan kepercayaan dan hubungan spiritual suku Asmat dengan leluhur.",
        "characteristics": [
            "Terinspirasi ukiran kayu Asmat",
            "Motif figur manusia dan makhluk spiritual",
            "Warna gelap: hitam, coklat, putih",
            "Simbolisme kepercayaan animisme",
            "Garis-garis kuat dan tegas",
        ],
    },
    "Papua_Cendrawasih": {
        "origin": "Papua",
        "region": "Papua",
        "description": "Batik Papua Cendrawasih menampilkan motif burung Cendrawasih (Bird of Paradise) yang merupakan burung ikonik Papua, melambangkan keindahan, kemerdekaan, dan keajaiban alam.",
        "characteristics": [
            "Motif burung Cendrawasih yang eksotis",
            "Warna cerah: merah, kuning, hijau",
            "Detail bulu yang mewah dan dramatis",
            "Simbol keindahan alam Papua",
            "Latar dengan motif hutan tropis",
        ],
    },
    "Papua_Tifa": {
        "origin": "Papua",
        "region": "Papua",
        "description": "Batik Papua Tifa menampilkan motif alat musik Tifa, drum tradisional Papua yang digunakan dalam upacara adat, melambangkan budaya dan tradisi masyarakat Papua.",
        "characteristics": [
            "Motif alat musik Tifa (gendang Papua)",
            "Warna-warna berani dan kontras",
            "Ornamen motif tradisional Papua",
            "Simbol budaya dan ritual adat",
            "Perpaduan dengan motif geometris Papua",
        ],
    },
    "Priangan_Merak_Ngibing": {
        "origin": "Jawa Barat",
        "region": "Priangan, Jawa Barat",
        "description": "Batik Priangan Merak Ngibing menampilkan motif burung merak yang sedang menari (ngibing), merupakan batik khas daerah Priangan Jawa Barat dengan nuansa elegan.",
        "characteristics": [
            "Motif merak menari (ngibing)",
            "Warna biru, hijau, dan emas",
            "Detail bulu merak yang sangat halus",
            "Karakter batik Sunda yang khas",
            "Melambangkan keanggunan dan keindahan",
        ],
    },
    "Sekar": {
        "origin": "Jawa Tengah",
        "region": "Solo dan Yogyakarta",
        "description": "Batik Sekar merupakan batik dengan motif bunga (sekar berarti bunga dalam bahasa Jawa), salah satu motif klasik keraton yang melambangkan keindahan dan kehalusan budi.",
        "characteristics": [
            "Motif bunga-bunga stilisasi",
            "Warna sogan (coklat keemasan)",
            "Motif klasik keraton Jawa",
            "Komposisi yang harmonis dan seimbang",
            "Melambangkan kehalusan dan keanggunan",
        ],
    },
    "Sidoluhur": {
        "origin": "Jawa Tengah",
        "region": "Solo, Jawa Tengah",
        "description": "Batik Sidoluhur adalah motif batik keraton Solo yang bermakna 'selalu luhur' atau senantiasa mulia, biasanya dipakai dalam upacara pernikahan adat Jawa.",
        "characteristics": [
            "Motif klasik keraton Solo",
            "Warna sogan coklat keemasan",
            "Bermakna keluhuran dan kemuliaan",
            "Digunakan dalam upacara pernikahan",
            "Motif bunga dan fauna stilisasi",
        ],
    },
    "Sogan": {
        "origin": "Jawa Tengah",
        "region": "Solo dan Yogyakarta",
        "description": "Batik Sogan adalah batik dengan pewarnaan sogan (coklat keemasan khas) yang dihasilkan dari kulit pohon soga. Merupakan gaya batik keraton klasik yang paling ikonik.",
        "characteristics": [
            "Warna coklat keemasan khas sogan",
            "Pewarna alami dari kulit pohon soga",
            "Motif keraton klasik Jawa",
            "Kontras hitam, putih, dan coklat",
            "Simbol budaya keraton tertinggi",
        ],
    },
    "Solo_Parang": {
        "origin": "Jawa Tengah",
        "region": "Solo, Jawa Tengah",
        "description": "Batik Solo Parang adalah motif parang khas keraton Surakarta, menampilkan pola diagonal yang melambangkan kekuatan, ketangkasan, dan keberanian pemimpinnya.",
        "characteristics": [
            "Pola diagonal berulang (parang)",
            "Warna sogan coklat dan hitam",
            "Motif larangan keraton Surakarta",
            "Melambangkan kekuatan dan ketangkasan",
            "Garis diagonal yang tegas dan ritmis",
        ],
    },
    "Sulawesi_Selatan_Lontara": {
        "origin": "Sulawesi Selatan",
        "region": "Makassar, Sulawesi Selatan",
        "description": "Batik Sulawesi Selatan Lontara menampilkan motif aksara Lontara, tulisan tradisional suku Bugis-Makassar yang menjadi keunikan tersendiri dalam dunia batik Indonesia.",
        "characteristics": [
            "Motif aksara Lontara Bugis-Makassar",
            "Warna merah, kuning, dan hitam",
            "Simbol literasi dan budaya Bugis",
            "Perpaduan geometris dan aksara",
            "Warisan budaya tulis Sulawesi Selatan",
        ],
    },
    "Sumatera_Barat_Rumah_Minang": {
        "origin": "Sumatera Barat",
        "region": "Padang, Sumatera Barat",
        "description": "Batik Sumatera Barat Rumah Minang menampilkan motif Rumah Gadang, rumah adat Minangkabau dengan atap tanduk kerbau yang ikonik, melambangkan kebudayaan Minang.",
        "characteristics": [
            "Motif Rumah Gadang dengan atap gonjong",
            "Warna emas, merah, dan hitam",
            "Simbol kebudayaan Minangkabau",
            "Ornamen ukiran Minang yang detail",
            "Melambangkan adat dan kebersamaan",
        ],
    },
    "Sumatera_Utara_Boraspati": {
        "origin": "Sumatera Utara",
        "region": "Batak, Sumatera Utara",
        "description": "Batik Sumatera Utara Boraspati menampilkan motif cicak/tokek (boraspati) yang merupakan simbol pelindung rumah dalam kepercayaan suku Batak Toba.",
        "characteristics": [
            "Motif cicak/tokek (boraspati ni huta)",
            "Warna merah, hitam, dan putih",
            "Simbol pelindung dan keberuntungan",
            "Pengaruh budaya Batak Toba",
            "Perpaduan motif ulos Batak",
        ],
    },
    "Tambal": {
        "origin": "Jawa Tengah",
        "region": "Jawa Tengah",
        "description": "Batik Tambal memiliki pola yang terdiri dari berbagai potongan motif berbeda yang disusun seperti tambalan, melambangkan keutuhan dari keberagaman dan dipercaya memiliki khasiat penyembuhan.",
        "characteristics": [
            "Pola tersusun dari aneka motif berbeda",
            "Tampak seperti kain tambalan",
            "Kombinasi berbagai warna",
            "Dipercaya memiliki khasiat penyembuhan",
            "Melambangkan keberagaman dalam kesatuan",
        ],
    },
    "Yogyakarta_Kawung": {
        "origin": "DI Yogyakarta",
        "region": "Yogyakarta",
        "description": "Batik Kawung adalah salah satu motif batik tertua dari Yogyakarta, berbentuk lingkaran-lingkaran yang berpotongan menyerupai buah kawung (kolang-kaling), melambangkan kesucian.",
        "characteristics": [
            "Pola lingkaran berpotongan (kawung)",
            "Warna hitam, putih, dan coklat sogan",
            "Salah satu motif batik tertua",
            "Motif larangan keraton Yogyakarta",
            "Melambangkan kesucian dan kemurnian",
        ],
    },
    "Yogyakarta_Parang": {
        "origin": "DI Yogyakarta",
        "region": "Yogyakarta",
        "description": "Batik Yogyakarta Parang merupakan motif parang khas Keraton Yogyakarta dengan pola diagonal yang kuat, melambangkan semangat pantang menyerah dan keteguhan hati.",
        "characteristics": [
            "Pola diagonal parang khas Yogyakarta",
            "Warna biru tua, hitam, dan putih",
            "Motif larangan keraton Yogyakarta",
            "Melambangkan semangat dan keteguhan",
            "Gaya lebih ramping dari parang Solo",
        ],
    },
}


class BatikPredictor:
    def __init__(self, model_path: str, labels_path: str, device: str):
        with open(labels_path, "rb") as f:
            label_data = pickle.load(f)
        self.idx_to_class: dict[int, str] = label_data["idx_to_class"]

        num_classes = len(self.idx_to_class)
        self.model = get_model(num_classes)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.model.to(device)

        self.device = device
        self.transform = get_transforms("val")

    def predict(self, image: Image.Image) -> list[dict]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top5 = torch.topk(probs, k=5)
        results = []
        for rank, (idx, prob) in enumerate(
            zip(top5.indices.tolist(), top5.values.tolist()), start=1
        ):
            results.append(
                {
                    "rank": rank,
                    "class": self.idx_to_class[idx],
                    "confidence": round(prob, 6),
                }
            )
        return results

    def get_class_info(self, class_name: str) -> dict:
        info = BATIK_INFO.get(class_name)
        if info is None:
            return {}
        return {"class_name": class_name, **info}
