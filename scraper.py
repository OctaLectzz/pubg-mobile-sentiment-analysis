"""
Scraper untuk mengambil review PUBG Mobile dari Apple App Store menggunakan RSS Feed.
"""
import os, pandas as pd, requests, time, random

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "reviews_pubg.csv")

COUNTRY_OPTIONS = {
    "🇮🇩 Indonesia": "id", "🇺🇸 United States": "us", "🇬🇧 United Kingdom": "gb",
    "🇯🇵 Japan": "jp", "🇰🇷 South Korea": "kr", "🇦🇺 Australia": "au",
    "🇮🇳 India": "in", "🇸🇬 Singapore": "sg", "🇲🇾 Malaysia": "my",
    "🇵🇭 Philippines": "ph", "🇹🇭 Thailand": "th", "🇧🇷 Brazil": "br",
    "🇩🇪 Germany": "de", "🇫🇷 France": "fr", "🇷🇺 Russia": "ru",
}

def label_sentiment(rating):
    if rating <= 2: return "Negatif"
    elif rating == 3: return "Netral"
    else: return "Positif"

def scrape_reviews(country="id", how_many=500, output_path=None):
    if output_path is None: output_path = OUTPUT_FILE
    if how_many > 500: how_many = 500
    num_pages = min((how_many // 50) + (1 if how_many % 50 > 0 else 0), 10)
    all_reviews = []
    for page in range(1, num_pages + 1):
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id=1330123889/sortby=mostrecent/json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200: continue
            entries = response.json().get('feed', {}).get('entry', [])
            if not entries: break
            for entry in entries:
                if 'im:rating' in entry:
                    all_reviews.append({
                        "review_text": entry.get('content', {}).get('label', ""),
                        "rating": int(entry.get('im:rating', {}).get('label', 0)),
                        "date": entry.get('updated', {}).get('label', "")[:10],
                        "title": entry.get('title', {}).get('label', ""),
                        "version": entry.get('im:version', {}).get('label', "Unknown"),
                        "username": entry.get('author', {}).get('name', {}).get('label', "Anonymous"),
                        "country": country.upper(),
                    })
            if len(all_reviews) >= how_many: break
            time.sleep(random.uniform(0.3, 0.8))
        except Exception as e:
            print(f"[ERROR] Page {page}: {e}"); break
    if not all_reviews: return pd.DataFrame()
    df = pd.DataFrame(all_reviews[:how_many])
    df = df.dropna(subset=["review_text"])
    df["sentiment"] = df["rating"].apply(label_sentiment)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df

def scrape_multiple_countries(countries, how_many_per_country=200, output_path=None):
    if output_path is None: output_path = OUTPUT_FILE
    all_dfs = []
    for country in countries:
        df = scrape_reviews(country=country, how_many=how_many_per_country, output_path=None)
        if not df.empty: all_dfs.append(df)
        time.sleep(random.uniform(0.5, 1.0))
    if not all_dfs: return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return combined

def generate_sample_data(output_path=None):
    if output_path is None: output_path = OUTPUT_FILE
    r = []
    # Positive reviews (rating 5)
    p5 = [
        ("Game terbaik! Grafisnya keren banget dan gameplay seru abis. Update terbaru makin mantap!", "Game Terbaik", "3.2.0"),
        ("PUBG Mobile battle royale paling seru! Main sejak awal tidak pernah bosan.", "Tidak Bosan", "3.2.0"),
        ("Mantap betul game ini, grafik HD sangat bagus. Map baru seru banget.", "Grafik Mantap", "3.2.0"),
        ("Suka banget update terbarunya! Mode baru menyenangkan dan bikin ketagihan.", "Update Bagus", "3.3.0"),
        ("Kontrol tembak smooth dan responsif. Recommended buat pecinta FPS!", "Kontrol Smooth", "3.3.0"),
        ("Best game ever! Setiap hari main terus gak bosan. TDM juga asik.", "Best Game", "3.3.0"),
        ("Grafiknya detail dan realistis. Efek suara senjata sangat bagus.", "Detail Bagus", "3.3.0"),
        ("PUBG terbaik! Setiap season content baru menarik.", "Content Menarik", "3.4.0"),
        ("Performa lancar jaya tanpa lag sedikitpun di hp saya.", "Lancar", "3.4.0"),
        ("Seru parah! Mabar sama temen makin asyik. Voice chat jernih.", "Seru Mabar", "3.4.0"),
        ("Versi baru lebih stabil dan grafis bagus. Terus ditingkatkan!", "Versi Baru Bagus", "3.5.0"),
        ("FPS lebih stabil dan loading lebih cepat dari sebelumnya!", "FPS Stabil", "3.5.0"),
        ("Sangat menikmati game ini bersama teman. Squad mode the best!", "Squad Best", "3.5.0"),
        ("Animasi karakter sangat halus dan detail. Puas mainnya!", "Animasi Halus", "3.4.0"),
        ("Event season ini keren banget! Hadiah bagus dan misi seru.", "Event Keren", "3.5.0"),
        ("Gameplay paling adiktif! Gak kerasa udah main 5 jam.", "Adiktif", "3.3.0"),
        ("Voice chat kualitasnya jernih. Koordinasi tim jadi mudah.", "Voice Chat Bagus", "3.4.0"),
        ("Map Erangel revamp sangat bagus! Detail lingkungan realistis.", "Map Bagus", "3.5.0"),
        ("Sistem ranking fair dan kompetitif. Seru push rank!", "Ranking Fair", "3.3.0"),
        ("Game battle royale mobile nomor satu! Tidak ada tandingannya.", "Nomor Satu", "3.2.0"),
    ]
    for txt, title, ver in p5:
        r.append({"review_text": txt, "rating": 5, "title": title, "version": ver})

    # Positive reviews (rating 4)
    p4 = [
        ("Game bagus overall, cuma kadang lag di mode 4v4. Tapi memuaskan.", "Bagus", "3.2.0"),
        ("Gameplay seru mapnya luas. Ukuran download agak besar. Perfect!", "Seru", "3.2.0"),
        ("Salah satu game terbaik di mobile. Kontrol bisa di-custom.", "Recommended", "3.2.0"),
        ("Lumayan bagus gamenya. Update terakhir ada improvement bagus.", "Lumayan Bagus", "3.3.0"),
        ("Game strategi yang bagus. Perlu teamwork untuk menang.", "Kompetitif", "3.3.0"),
        ("Senang game ini, banyak mode bisa dimainkan. Storage agak banyak.", "Banyak Mode", "3.3.0"),
        ("Game solid dan stabil. Jarang crash sekarang.", "Stabil", "3.3.0"),
        ("Update terbaru bagus, senjata baru dan kendaraan baru. Seru!", "Senjata Baru", "3.4.0"),
        ("Grafisnya memukau untuk game mobile. Anti aliasing smooth.", "Grafis Memukau", "3.4.0"),
        ("Game FPS mobile terbaik. Matchmaking lumayan cepat dan fair.", "FPS Terbaik", "3.4.0"),
        ("Fitur baru keren tapi ukuran game makin membengkak.", "Fitur Keren", "3.5.0"),
        ("Setelah maintenance game jadi lebih lancar. Terima kasih dev!", "Lebih Lancar", "3.5.0"),
        ("Gameplay cukup memuaskan walau ada beberapa bug kecil.", "Cukup Puas", "3.4.0"),
        ("Senjata baru balanced dan seru dipakai. Good job dev!", "Balanced", "3.5.0"),
        ("Pengalaman bermain sangat menyenangkan bersama squad.", "Menyenangkan", "3.3.0"),
        ("Performa di HP mid-range sudah lumayan baik sekarang.", "Performa OK", "3.4.0"),
        ("Training mode sangat membantu untuk latihan aim.", "Training Bagus", "3.2.0"),
        ("UI baru lebih clean dan mudah dinavigasi.", "UI Bagus", "3.5.0"),
        ("Matchmaking cepat dan jarang ketemu cheater sekarang.", "Match Cepat", "3.4.0"),
        ("Game terus berkembang dengan konten baru setiap bulan.", "Berkembang", "3.3.0"),
    ]
    for txt, title, ver in p4:
        r.append({"review_text": txt, "rating": 4, "title": title, "version": ver})

    # Neutral reviews (rating 3)
    n3 = [
        ("Game biasa aja, kadang seru kadang kesel karena cheater.", "Biasa Aja", "3.2.0"),
        ("Lumayan buat mengisi waktu. Terlalu banyak bot di tier rendah.", "Lumayan", "3.2.0"),
        ("Game OK tapi butuh HP spek tinggi. HP kentang ngelag.", "Butuh HP Bagus", "3.3.0"),
        ("Bagus gamenya tapi matchmaking gak balance. Newbie diadu pro.", "Matchmaking Jelek", "3.3.0"),
        ("Standard battle royale. Gak istimewa tapi gak jelek juga.", "Standard", "3.3.0"),
        ("Oke gamenya, server suka maintenance mendadak. Ganggu.", "Maintenance", "3.3.0"),
        ("Fifty-fifty, kadang lancar kadang lag parah.", "Fifty-fifty", "3.4.0"),
        ("Cukup menyenangkan tapi update terakhir ada bug mengganggu.", "Ada Bug", "3.4.0"),
        ("Plus minusnya. Plusnya seru, minusnya banyak in-app purchase.", "Plus Minus", "3.4.0"),
        ("Gamenya bagus tapi kurang optimized untuk beberapa device.", "Kurang Optimized", "3.4.0"),
        ("Lumayan versi baru, ada perbaikan bug. Masih ada lag.", "Lumayan", "3.5.0"),
        ("Game masih oke, event kurang menarik season ini.", "Event Kurang", "3.5.0"),
        ("Tidak buruk tapi juga tidak istimewa. Biasa saja.", "Biasa", "3.2.0"),
        ("Kadang seru kadang bosan tergantung mood.", "Tergantung Mood", "3.3.0"),
        ("Game lumayan tapi terlalu banyak iklan dan promosi skin.", "Banyak Iklan", "3.4.0"),
        ("Cukup menghibur tapi perlu perbaikan di anti cheat.", "Cukup Hibur", "3.5.0"),
        ("Gameplay oke tapi grafis di HP low-end jelek banget.", "Grafis Low", "3.3.0"),
        ("Update ada bagusnya ada jeleknya. Mixed feelings.", "Mixed", "3.4.0"),
        ("Game besar tapi kontennya tidak sebanding ukurannya.", "Konten Kurang", "3.5.0"),
        ("Rata-rata saja, banyak game sejenis yang lebih baik.", "Rata-rata", "3.2.0"),
    ]
    for txt, title, ver in n3:
        r.append({"review_text": txt, "rating": 3, "title": title, "version": ver})

    # Negative reviews (rating 2)
    n2 = [
        ("Game makin berat. Update bikin hp panas dan boros baterai.", "HP Panas", "3.2.0"),
        ("Banyak cheater dan hacker! Developer gak becus. Mengecewakan.", "Banyak Cheater", "3.2.0"),
        ("Lag parah sejak update. HP flagship tetep ngelag.", "Lag Parah", "3.2.0"),
        ("Pay to win. Skin mahal gameplay jelek. Matchmaking tidak adil.", "Pay To Win", "3.3.0"),
        ("Sering crash dan force close. Reinstall masalah tetap sama.", "Sering Crash", "3.3.0"),
        ("Update bikin game tambah lemot. Optimasi buruk mid-range.", "Makin Lemot", "3.3.0"),
        ("Server Indonesia sering down. Ping tinggi terus.", "Server Down", "3.3.0"),
        ("Kecewa, dulu bagus sekarang banyak masalah. Bug dimana-mana.", "Kecewa", "3.4.0"),
        ("Audio glitch suara senjata hilang saat pertarungan. Mengganggu.", "Audio Glitch", "3.4.0"),
        ("Anti cheat payah. Banyak aim bot dan wall hack.", "Anti Cheat Buruk", "3.4.0"),
        ("Update versi baru control berubah aneh. Sensitivitas berubah.", "Control Aneh", "3.5.0"),
        ("Tidak bagus lagi, game ini makin menurun kualitasnya.", "Menurun", "3.2.0"),
        ("Matchmaking sangat tidak adil. Selalu kalah.", "Tidak Adil", "3.3.0"),
        ("Loading lama banget, bisa 5 menit baru masuk game.", "Loading Lama", "3.4.0"),
        ("Desync parah! Sudah tembak duluan tapi yang mati saya.", "Desync", "3.5.0"),
        ("Grafis menurun setelah update. Tekstur jadi blur.", "Grafis Turun", "3.3.0"),
        ("Game tidak bisa dimainkan tanpa internet yang stabil.", "Internet Issue", "3.4.0"),
        ("Skin terlalu mahal dan tidak worth it.", "Skin Mahal", "3.5.0"),
        ("Sistem report cheater tidak efektif sama sekali.", "Report Gagal", "3.2.0"),
        ("Game tidak menyenangkan kalau solo. Harus punya squad.", "Solo Boring", "3.3.0"),
    ]
    for txt, title, ver in n2:
        r.append({"review_text": txt, "rating": 2, "title": title, "version": ver})

    # Very negative reviews (rating 1)
    n1 = [
        ("SAMPAH! Game terburuk. Penuh cheater dan bug!", "Sampah", "3.2.0"),
        ("Uninstall! Tidak layak dimainkan. Server jelek cheater merajalela.", "Uninstall", "3.2.0"),
        ("Sangat mengecewakan! Setiap update makin parah.", "Mengecewakan", "3.2.0"),
        ("Cuma menguras uang! Skin mahal gameplay jelek banyak bug.", "Menguras Uang", "3.3.0"),
        ("TERBURUK! Lag crash cheater bug semua ada!", "Terburuk", "3.3.0"),
        ("Game sudah mati! Player tinggal bot. Matchmaking lama.", "Game Mati", "3.3.0"),
        ("Tidak bisa login 3 hari! CS tidak membantu.", "Login Error", "3.3.0"),
        ("Akun kena ban tanpa alasan! Tidak pernah cheat. CURANG!", "Kena Ban", "3.4.0"),
        ("Paling boros kuota dan baterai. 1 jam baterai habis 50%.", "Boros", "3.4.0"),
        ("HAPUS GAME INI! Penuh masalah developer tidak kompeten!", "Hapus", "3.4.0"),
        ("Versi ini paling buruk! Lag dan sering disconnect ranked.", "Terburuk", "3.5.0"),
        ("Setiap update selalu masalah baru. Capek bug terus!", "Selalu Bug", "3.5.0"),
        ("Game tidak seru sama sekali. Membosankan!", "Membosankan", "3.2.0"),
        ("Developer hanya peduli uang, tidak peduli player!", "Greedy Dev", "3.3.0"),
        ("Worst game! Rugi sudah download 3GB.", "Worst Game", "3.4.0"),
        ("Hancur! Game rusak setelah update terakhir.", "Hancur", "3.5.0"),
        ("Tidak recommended! Buang waktu dan kuota.", "No Recommend", "3.2.0"),
        ("Customer service tidak responsif. Masalah tidak diselesaikan.", "CS Buruk", "3.3.0"),
        ("Game penuh ketidakadilan. Sistem ranking rusak.", "Tidak Adil", "3.4.0"),
        ("Sangat buruk! Menyesal sudah main game ini.", "Menyesal", "3.5.0"),
    ]
    for txt, title, ver in n1:
        r.append({"review_text": txt, "rating": 1, "title": title, "version": ver})

    # Add metadata
    import random as rnd
    dates = [f"2025-{m:02d}-{d:02d}" for m in range(1, 4) for d in range(1, 29)]
    for i, review in enumerate(r):
        review["date"] = rnd.choice(dates)
        review["username"] = f"user{i+1:03d}"
        review["country"] = "ID"

    df = pd.DataFrame(r)
    df["sentiment"] = df["rating"].apply(label_sentiment)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Sample data: {len(df)} review -> {output_path}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scraper Review PUBG Mobile")
    parser.add_argument("--country", default="id")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    if args.sample: generate_sample_data()
    else: scrape_reviews(country=args.country, how_many=args.count)
