import json
import random

def generate_dataset():
    # 400 Positif (Apresiasi, Harapan, Dukungan)
    pos_templates = [
        "alhamdulillah menu mbg hari ini enak bgt, anakku nambah terus",
        "mantap bgn! gizi anak bangsa emang nomor satu",
        "makasih pak presiden, program mbg beneran bantu rakyat kecil",
        "menyala abangku! sdm indonesia auto cerdas kalo begini",
        "seneng liat menu sppg hari ini lengkap, ada buah dan susu",
        "mbg sangat membantu ringankan beban dapur emak-emak",
        "terimakasih bgn sudah perhatikan gizi anak pelosok",
        "semangat para relawan dapur mbg, pahlawan tanpa tanda jasa",
        "keren sih menunya ganti terus tiap hari jadi bocah gak bosen",
        "porsinya pas gizinya lengkap masyaallah berkah bgt",
        "indonesia makin kuat kalo gizinya diperhatiin gini",
        "anakku jadi doyan sayur gara gara ikut program mbg, top!",
        "mantap betul pengelolaan sppg di daerahku profesional bgt",
        "semoga mbg jadi program permanen selamanya, keren abis",
        "gizi terjaga belajar jadi fokus, makasih mbg!",
        "bangga bgt sama program ini, nyata manfaatnya",
        "masyaallah tabarakallah anak sekolah dapet gizi bagus tiap hari",
        "program mbg emang solusi stunting paling oke",
        "makasih tim bgn sudah kerja keras buat anak sekolah",
        "sukses terus mbg, masa depan anak bangsa makin cerah"
    ]

    # 400 Negatif (Gaji, Korupsi, Makanan Basi, AI)
    neg_templates = [
        "gaji sppi batch 3 mana woy? udah tanggal segini cuma janji doang",
        "parah bgt menunya cuma nasi tempe, dikorupsi ya anggarannya?",
        "anak tetangga keracunan mbg kemarin, qc diperketat dong!",
        "percuma program gede kalo gaji pegawainya nunggak terus, dzolim",
        "bau-bau korupsi kerasa bgt, menunya gak layak buat anak",
        "admin bgn pengecut, nagih gaji malah komen dihapusin mulu",
        "pake ai mulu kontennya padahal realitanya zonk abis",
        "katanya tanggal 1 gajian tapi ampe sekarang hilal gak ada",
        "mending bubarin aja mbg kalo cuma bikin harga ayam naik terus",
        "dapur mbg di pemukiman ganggu bgt, berisik dan bau sampah",
        "kerja overtime tiap hari tapi bayaran gak jelas, kapok ikut sppi",
        "kenapa harus belajar ke india? emang gak ada negara lain?",
        "proyek nepotisme, yg dapet kerja cuma saudara orang dalam doang",
        "makanan basi dikasih ke anak kecil, gak punya hati nurani ya?",
        "kecewa bgt sistem bgn berantakan, manajemennya payah parah",
        "janji manis doang gaji masuk tanggal 1, nyatanya zonk",
        "anggaran triliunan tapi bayar gaji sppi aja susah bgt",
        "program dipaksa-paksa tapi operasionalnya acakadut",
        "viralin gaji nunggak! jangan mau dibohongi terus sama oknum bgn",
        "menu harganya 15rb tapi aslinya cuma kayak 5rb, sisanya kemana?"
    ]

    # 400 Netral (Info, Tanya, Loker, Teknis)
    neu_templates = [
        "min cara daftar jadi supplier telur di sppg gimana ya?",
        "cek dm min ada penawaran kerjasama dari umkm catering lokal",
        "untuk wilayah kalimantan barat udah mulai jalan belum ya programnya?",
        "halo badangizi mau tanya kalo mau resign prosedurnya gimana?",
        "info loker ahli gizi buat daerah jawa tengah dong min",
        "ini telepon pengaduan bgn berbayar atau gratis ya min?",
        "apa benar syarat jadi sppi harus lulusan sarjana gizi?",
        "lokasi dapur mbg di jakarta selatan ada di mana aja ya?",
        "mohon pencerahan buat alur distribusi ke sekolah swasta",
        "link dashboard mbg lagi down ya gak bisa login dari pagi",
        "apakah ada sosialisasi buat orang tua murid di sekolah?",
        "min tolong spill menu lengkap mbg buat minggu ini",
        "saya mau lapor tapi lewat wa, ada nomornya gak ya?",
        "bagaimana mekanisme pengawasan buat dapur sppg mandiri?",
        "info jadwal pencairan banper buat bulan depan min",
        "apakah sekolah luar biasa juga dapet program mbg ini?",
        "minimal pendidikan buat admin di sppg apa ya?",
        "cara cek status pendaftaran sppi gimana min?",
        "mau tanya apakah susu mbg ini mengandung pemanis buatan?",
        "tadi ada kurir bgn lewat depan rumah nanya alamat sekolah"
    ]

    dataset = []
    
    # Generate 400 per kategori dengan sedikit variasi tambahan
    for i in range(400):
        dataset.append({"text": f"{random.choice(pos_templates)} #{i+1}", "label": 2})
        dataset.append({"text": f"{random.choice(neg_templates)} #{i+1}", "label": 0})
        dataset.append({"text": f"{random.choice(neu_templates)} #{i+1}", "label": 1})

    random.shuffle(dataset) # Acak urutannya

    with open('dataset_mbg_1200.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("Selesai! File 'dataset_mbg_1200.json' telah berhasil dibuat.")

generate_dataset()