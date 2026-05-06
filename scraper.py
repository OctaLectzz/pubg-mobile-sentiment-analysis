"""
Scraper untuk mengambil review PUBG Mobile dari Apple App Store.
"""
import os, sys, pandas as pd

try:
    from app_store_scraper import AppStore
except ImportError:
    AppStore = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "reviews_pubg.csv")

def label_sentiment(rating):
    if rating <= 2: return "Negatif"
    elif rating == 3: return "Netral"
    else: return "Positif"

def scrape_reviews(country="id", how_many=2000, output_path=None):
    if output_path is None: output_path = OUTPUT_FILE
    print(f"[INFO] Scraping {how_many} review PUBG Mobile ({country.upper()})...")
    try:
        app = AppStore(country=country, app_name="pubg-mobile", app_id=1330123889)
        app.review(how_many=how_many)
        if not app.reviews:
            print("[WARNING] Tidak ada review ditemukan."); return pd.DataFrame()
        df = pd.DataFrame(app.reviews)
        col_map = {"review":"review_text","userName":"username","rating":"rating","date":"date","title":"title"}
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
        for vc in ["version","appVersion","app_version"]:
            if vc in df.columns:
                df = df.rename(columns={vc:"version"}); break
        if "version" not in df.columns: df["version"] = "Unknown"
        for c in ["review_text","rating","date","title","version","username"]:
            if c not in df.columns: df[c] = ""
        df = df[["review_text","rating","date","title","version","username"]]
        df = df.dropna(subset=["review_text"])
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce"); df = df.dropna(subset=["rating"])
        df["rating"] = df["rating"].astype(int)
        df["sentiment"] = df["rating"].apply(label_sentiment)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[SUCCESS] {len(df)} review disimpan ke {output_path}")
        return df
    except Exception as e:
        print(f"[ERROR] Gagal: {e}"); return pd.DataFrame()

def generate_sample_data(output_path=None):
    if output_path is None: output_path = OUTPUT_FILE
    reviews = [
        {"review_text":"Game terbaik! Grafisnya keren banget dan gameplay seru abis. Update terbaru makin mantap!","rating":5,"date":"2025-01-15","title":"Game Terbaik","version":"3.2.0","username":"user001"},
        {"review_text":"PUBG Mobile battle royale paling seru! Main sejak awal tidak pernah bosan.","rating":5,"date":"2025-01-20","title":"Tidak Bosan","version":"3.2.0","username":"user002"},
        {"review_text":"Mantap betul game ini, grafik HD sangat bagus. Map baru seru banget.","rating":5,"date":"2025-02-01","title":"Grafik Mantap","version":"3.2.0","username":"user003"},
        {"review_text":"Suka banget update terbarunya! Mode baru menyenangkan dan bikin ketagihan.","rating":5,"date":"2025-02-05","title":"Update Bagus","version":"3.3.0","username":"user004"},
        {"review_text":"Kontrol tembak smooth dan responsif. Recommended buat pecinta FPS!","rating":5,"date":"2025-02-10","title":"Kontrol Smooth","version":"3.3.0","username":"user005"},
        {"review_text":"Best game ever! Setiap hari main terus gak bosan. TDM juga asik.","rating":5,"date":"2025-02-15","title":"Best Game","version":"3.3.0","username":"user006"},
        {"review_text":"Grafiknya detail dan realistis. Efek suara senjata sangat bagus.","rating":5,"date":"2025-03-01","title":"Detail Bagus","version":"3.3.0","username":"user007"},
        {"review_text":"PUBG terbaik! Setiap season content baru menarik.","rating":5,"date":"2025-03-05","title":"Content Menarik","version":"3.4.0","username":"user008"},
        {"review_text":"Performa lancar jaya tanpa lag sedikitpun di hp saya.","rating":5,"date":"2025-03-10","title":"Lancar","version":"3.4.0","username":"user009"},
        {"review_text":"Seru parah! Mabar sama temen makin asyik. Voice chat jernih.","rating":5,"date":"2025-03-15","title":"Seru Mabar","version":"3.4.0","username":"user010"},
        {"review_text":"Game bagus overall, cuma kadang lag di mode 4v4. Tapi memuaskan.","rating":4,"date":"2025-01-18","title":"Bagus","version":"3.2.0","username":"user011"},
        {"review_text":"Gameplay seru mapnya luas. Ukuran download agak besar. Perfect!","rating":4,"date":"2025-01-25","title":"Seru","version":"3.2.0","username":"user012"},
        {"review_text":"Salah satu game terbaik di mobile. Kontrol bisa di-custom.","rating":4,"date":"2025-02-03","title":"Recommended","version":"3.2.0","username":"user013"},
        {"review_text":"Lumayan bagus gamenya. Update terakhir ada improvement bagus.","rating":4,"date":"2025-02-08","title":"Lumayan Bagus","version":"3.3.0","username":"user014"},
        {"review_text":"Game strategi yang bagus. Perlu teamwork untuk menang.","rating":4,"date":"2025-02-12","title":"Kompetitif","version":"3.3.0","username":"user015"},
        {"review_text":"Senang game ini, banyak mode bisa dimainkan. Storage agak banyak.","rating":4,"date":"2025-02-20","title":"Banyak Mode","version":"3.3.0","username":"user016"},
        {"review_text":"Game solid dan stabil. Jarang crash sekarang.","rating":4,"date":"2025-03-02","title":"Stabil","version":"3.3.0","username":"user017"},
        {"review_text":"Update terbaru bagus, senjata baru dan kendaraan baru. Seru!","rating":4,"date":"2025-03-07","title":"Senjata Baru","version":"3.4.0","username":"user018"},
        {"review_text":"Grafisnya memukau untuk game mobile. Anti aliasing smooth.","rating":4,"date":"2025-03-12","title":"Grafis Memukau","version":"3.4.0","username":"user019"},
        {"review_text":"Game FPS mobile terbaik. Matchmaking lumayan cepat dan fair.","rating":4,"date":"2025-03-18","title":"FPS Terbaik","version":"3.4.0","username":"user020"},
        {"review_text":"Game biasa aja, kadang seru kadang kesel karena cheater.","rating":3,"date":"2025-01-22","title":"Biasa Aja","version":"3.2.0","username":"user021"},
        {"review_text":"Lumayan buat mengisi waktu. Terlalu banyak bot di tier rendah.","rating":3,"date":"2025-01-28","title":"Lumayan","version":"3.2.0","username":"user022"},
        {"review_text":"Game OK tapi butuh HP spek tinggi. HP kentang ngelag.","rating":3,"date":"2025-02-06","title":"Butuh HP Bagus","version":"3.3.0","username":"user023"},
        {"review_text":"Bagus gamenya tapi matchmaking gak balance. Newbie diadu pro.","rating":3,"date":"2025-02-14","title":"Matchmaking Jelek","version":"3.3.0","username":"user024"},
        {"review_text":"Standard battle royale. Gak istimewa tapi gak jelek juga.","rating":3,"date":"2025-02-22","title":"Standard","version":"3.3.0","username":"user025"},
        {"review_text":"Oke gamenya, server suka maintenance mendadak. Ganggu.","rating":3,"date":"2025-03-03","title":"Maintenance","version":"3.3.0","username":"user026"},
        {"review_text":"Fifty-fifty, kadang lancar kadang lag parah.","rating":3,"date":"2025-03-08","title":"Fifty-fifty","version":"3.4.0","username":"user027"},
        {"review_text":"Cukup menyenangkan tapi update terakhir ada bug mengganggu.","rating":3,"date":"2025-03-13","title":"Ada Bug","version":"3.4.0","username":"user028"},
        {"review_text":"Plus minusnya. Plusnya seru, minusnya banyak in-app purchase.","rating":3,"date":"2025-03-20","title":"Plus Minus","version":"3.4.0","username":"user029"},
        {"review_text":"Gamenya bagus tapi kurang optimized untuk beberapa device.","rating":3,"date":"2025-03-25","title":"Kurang Optimized","version":"3.4.0","username":"user030"},
        {"review_text":"Game makin berat. Update bikin hp panas dan boros baterai.","rating":2,"date":"2025-01-19","title":"HP Panas","version":"3.2.0","username":"user031"},
        {"review_text":"Banyak cheater dan hacker! Developer gak becus. Mengecewakan.","rating":2,"date":"2025-01-26","title":"Banyak Cheater","version":"3.2.0","username":"user032"},
        {"review_text":"Lag parah sejak update. HP flagship tetep ngelag.","rating":2,"date":"2025-02-04","title":"Lag Parah","version":"3.2.0","username":"user033"},
        {"review_text":"Pay to win. Skin mahal gameplay jelek. Matchmaking tidak adil.","rating":2,"date":"2025-02-09","title":"Pay To Win","version":"3.3.0","username":"user034"},
        {"review_text":"Sering crash dan force close. Reinstall masalah tetap sama.","rating":2,"date":"2025-02-16","title":"Sering Crash","version":"3.3.0","username":"user035"},
        {"review_text":"Update bikin game tambah lemot. Optimasi buruk mid-range.","rating":2,"date":"2025-02-24","title":"Makin Lemot","version":"3.3.0","username":"user036"},
        {"review_text":"Server Indonesia sering down. Ping tinggi terus.","rating":2,"date":"2025-03-04","title":"Server Down","version":"3.3.0","username":"user037"},
        {"review_text":"Kecewa, dulu bagus sekarang banyak masalah. Bug dimana-mana.","rating":2,"date":"2025-03-09","title":"Kecewa","version":"3.4.0","username":"user038"},
        {"review_text":"Audio glitch suara senjata hilang saat pertarungan. Mengganggu.","rating":2,"date":"2025-03-14","title":"Audio Glitch","version":"3.4.0","username":"user039"},
        {"review_text":"Anti cheat payah. Banyak aim bot dan wall hack.","rating":2,"date":"2025-03-19","title":"Anti Cheat Buruk","version":"3.4.0","username":"user040"},
        {"review_text":"SAMPAH! Game terburuk. Penuh cheater dan bug!","rating":1,"date":"2025-01-16","title":"Sampah","version":"3.2.0","username":"user041"},
        {"review_text":"Uninstall! Tidak layak dimainkan. Server jelek cheater merajalela.","rating":1,"date":"2025-01-23","title":"Uninstall","version":"3.2.0","username":"user042"},
        {"review_text":"Sangat mengecewakan! Setiap update makin parah.","rating":1,"date":"2025-01-30","title":"Mengecewakan","version":"3.2.0","username":"user043"},
        {"review_text":"Cuma menguras uang! Skin mahal gameplay jelek banyak bug.","rating":1,"date":"2025-02-07","title":"Menguras Uang","version":"3.3.0","username":"user044"},
        {"review_text":"TERBURUK! Lag crash cheater bug semua ada!","rating":1,"date":"2025-02-13","title":"Terburuk","version":"3.3.0","username":"user045"},
        {"review_text":"Game sudah mati! Player tinggal bot. Matchmaking lama.","rating":1,"date":"2025-02-21","title":"Game Mati","version":"3.3.0","username":"user046"},
        {"review_text":"Tidak bisa login 3 hari! CS tidak membantu.","rating":1,"date":"2025-03-01","title":"Login Error","version":"3.3.0","username":"user047"},
        {"review_text":"Akun kena ban tanpa alasan! Tidak pernah cheat. CURANG!","rating":1,"date":"2025-03-06","title":"Kena Ban","version":"3.4.0","username":"user048"},
        {"review_text":"Paling boros kuota dan baterai. 1 jam baterai habis 50%.","rating":1,"date":"2025-03-11","title":"Boros","version":"3.4.0","username":"user049"},
        {"review_text":"HAPUS GAME INI! Penuh masalah developer tidak kompeten!","rating":1,"date":"2025-03-16","title":"Hapus","version":"3.4.0","username":"user050"},
        {"review_text":"Update versi baru control berubah aneh. Sensitivitas berubah.","rating":2,"date":"2025-03-22","title":"Control Aneh","version":"3.5.0","username":"user051"},
        {"review_text":"Versi baru lebih stabil dan grafis bagus. Terus ditingkatkan!","rating":5,"date":"2025-03-23","title":"Versi Baru Bagus","version":"3.5.0","username":"user052"},
        {"review_text":"Lumayan versi baru, ada perbaikan bug. Masih ada lag.","rating":3,"date":"2025-03-24","title":"Lumayan","version":"3.5.0","username":"user053"},
        {"review_text":"FPS lebih stabil dan loading lebih cepat dari sebelumnya!","rating":5,"date":"2025-03-25","title":"FPS Stabil","version":"3.5.0","username":"user054"},
        {"review_text":"Versi ini paling buruk! Lag dan sering disconnect ranked.","rating":1,"date":"2025-03-26","title":"Terburuk","version":"3.5.0","username":"user055"},
        {"review_text":"Fitur baru keren tapi ukuran game makin membengkak.","rating":4,"date":"2025-03-27","title":"Fitur Keren","version":"3.5.0","username":"user056"},
        {"review_text":"Setelah maintenance game jadi lebih lancar. Terima kasih dev!","rating":4,"date":"2025-03-28","title":"Lebih Lancar","version":"3.5.0","username":"user057"},
        {"review_text":"Game masih oke, event kurang menarik season ini.","rating":3,"date":"2025-03-29","title":"Event Kurang","version":"3.5.0","username":"user058"},
        {"review_text":"Setiap update selalu masalah baru. Capek bug terus!","rating":1,"date":"2025-03-30","title":"Selalu Bug","version":"3.5.0","username":"user059"},
        {"review_text":"Sangat menikmati game ini bersama teman. Squad mode the best!","rating":5,"date":"2025-03-31","title":"Squad Best","version":"3.5.0","username":"user060"},
    ]
    df = pd.DataFrame(reviews)
    df["sentiment"] = df["rating"].apply(label_sentiment)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Sample data: {len(df)} review -> {output_path}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scraper Review PUBG Mobile")
    parser.add_argument("--country", default="id")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    if args.sample: generate_sample_data()
    else: scrape_reviews(country=args.country, how_many=args.count)
