class Prompts():

    STORY_GENRE = ['fantasy', 'mystery', 'adventure', 'science', 'fairy tale', 'fable', 'horror', 'humor', 'realistic', 'historical', 'sports', 'superhero', 'family', 'friendship']
    
    STORY_WORD_LENGTH = [50, 100, 150, 200]

    STORY_PROMPT = '''
        Write one short fiction children story about a {STORY_GENRE} setting in Bahasa Indonesia.
        The story should be suitable for children and should not contain any harmful content.
        The story should be written in {WORD_LENGTH} words.
        The story should be written in a simple and easy to understand language.
        The sroty should be suitable to be used as a context for a elementary grade student reading comprehension test.

        Here is an example of the story:

        "Hari ini adalah hari ulang tahunku. Pagi-pagi ayah dan ibu sudah datang ke kamar untuk membangunkanku seraya mengucapkan selamat ulang tahun. Kemudian ibu mencium dan memelukku dengan erat. Ayah berjanji bahwa sepulang sekolah akan memberikanku hadiah. Aku sangat penasaran akan hadiah tersebut hingga membuatku semangat untuk menjalani hari. Sepulang sekolah, ayahku mengajakku untuk pergi ke toko sepeda dan membelikanku sebuah sepeda baru. Aku sangat senang akan hadiah yang diberikan oleh ayah."

        "Si Lancang sudah mulai bosan dengan kehidupan yang serba kekurangan. Ia mengeluh, tampak putus asa. Berkali-kali ibunya memberi nasihat kepada Si Lancang agar anaknya tekun bekerja. “Sabarlah, Nak! Janganlah kamu terus-terusan mengeluh! Kita memang harus bekerja keras untuk memenuhi kebutuhan hidup sehari-hari. Jangan putus asa dan jangan menyerah. “Begitu ibu Si Lancang menasihati anak semata wayangnya itu."

        "Malin kundang turun dari kapal. Ia disambut oleh ibunya. Setelah cukup dekat, ibunya melihat bekas luka di lengan kanannya yang membuat ibu tersebut yakin bahwa ia adalah anaknya. “Malin Kundang, anakku, mengapa kau pergi begitu lama tanpa mengirimkan kabar?”, kata ibu tersebut sambil memeluk Malin Kundang. Kemudian Malin Kundang langsung melepaskan pelukan ibunya dan mendorongnya hingga terjatuh. “Wanita tak tahu diri, sembarangan saja mengaku sebagai ibuku”, kata Malin Kundang pada ibunya. Malin Kundang pura-pura tidak mengenali ibunya, karena malu dengan ibunya yang sudah tua dan memakai baju yang compang-camping."

        "Kemarin ibu guru memberitahukan kepada kami bahwa hari ini akan ada pemberian vitamin. Kami diminta membawa surat permohonan izin orang tua untuk ditandatangani. Ibu guru mengatakan jika disuntik hanya sakit sebentar lalu kami akan kembali ceria. Hari ini tiba saatnya kami disuntik. Ani menangis ketakutan. Budi berlarian kesana kemari. Aku ingat pesan ayah bahwa aku harus menjadi anak yang pemberani. Akhirnya aku disuntik tanpa menangis. Ibu guru mengatakan bahwa aku anak yang hebat."

        "Ayah selaku bangun pagi. Setelah bersiap-siap, dia berjalan tanpa henti. Dengan sigap aku berlari mengikuti. Di sawah, kami saling bantu. Ayah mencangkul tanah berlumpur. Aku mencabuti rumput di pesemaian. Ayahku petani. Lengannya kuat, berkilat bagai besi. Senyumnya seterang matahari."

        "Pukul 09.00 aku dan Ibu sampai di stasiun tujuan. Ketika Ibu menata barang bawaan, aku meminta izin ke kamar kecil. Kuikuti petunjuk arah berupa tanda panah dan tulisan “Toilet”. Oh, ternyata aku harus berbelok beberapa kali. Setelah selesai buang air kecil, aku mencuci tangan dan segera kembali. Kuingat-ingat belokan yang tadi kulewati. Aku menoleh ke kanan dan kiri, tetapi Ibu tidak terlihat. Bagaimana ini? Di mana aku? Dadaku mulai berdebar-debar. Mungkin aku salah belok, mungkin aku tersesat!"

        "Aku melangkah lebih cepat, kugenggam tangan Ibu erat-erat. Di sekelilingku, orang-orang berjalan tergesa-gesa. Mereka terus-menerus melihat arloji atau papan pengumuman. “Ayo, Nina. Ini gerbong kita,” ajak Ibu. Kami segera naik, mencari nomor kursi sesuai tiket, lalu duduk. Tak lama kemudian, aku merasakan hentakan perlahan. Kereta mulai bergerak ... kian cepat ... kian cepat .... Aku dan Ibu tersenyum lega karena kami tidak terlambat"
    '''.strip()

    QUESTION_ANSWER_PROMPT = '''
        Write multiple question and short answer about the story below delimited by the triple backtick (```) symbol:
        
        ```{STORY}```

        The question and answer should be written in Bahasa Indonesia.
        The question should be suitable for elementary grade student reading comprehension test.
        The answer should be a short answer that can be found in the story.
        The answer should be extracted from the story and should be exactly the same as the story, do not change the capitalization and the arrangement.
        The minimum number of question and answer is 5 and the maximum is 10.

        Your response should be in JSON format with the following keys: question, answer.

        
        Here is an example of the input and output:

        Input:
        Malin kundang turun dari kapal. Ia disambut oleh ibunya. Setelah cukup dekat, ibunya melihat bekas luka di lengan kanannya yang membuat ibu tersebut yakin bahwa ia adalah anaknya. “Malin Kundang, anakku, mengapa kau pergi begitu lama tanpa mengirimkan kabar?”, kata ibu tersebut sambil memeluk Malin Kundang. Kemudian Malin Kundang langsung melepaskan pelukan ibunya dan mendorongnya hingga terjatuh. “Wanita tak tahu diri, sembarangan saja mengaku sebagai ibuku”, kata Malin Kundang pada ibunya. Malin Kundang pura-pura tidak mengenali ibunya, karena malu dengan ibunya yang sudah tua dan memakai baju yang compang-camping.

        Output:
        [
            {
                "question": "Dari mana Malin Kundang turun?",
                "answer": "kapal"
            },
            {
                "question": "Bagaimana ibunya yakin bahwa Malin Kundang adalah anaknya?",
                "answer": "ibunya melihat bekas luka di lengan kanannya"
            },
            {
                "question": "Apa yang dilakukan Malin Kundang ketika ibunya memeluk Malin Kundang?",
                "answer": "melepaskan pelukan ibunya dan mendorongnya hingga terjatuh"
            },
            {
                "question": "Mengapa Malin Kundang pura-pura tidak mengenali ibunya?",
                "answer": "malu"
            }
        ]

        
        Input:
        Pukul 09.00 aku dan Ibu sampai di stasiun tujuan. Ketika Ibu menata barang bawaan, aku meminta izin ke kamar kecil. Kuikuti petunjuk arah berupa tanda panah dan tulisan “Toilet”. Oh, ternyata aku harus berbelok beberapa kali. Setelah selesai buang air kecil, aku mencuci tangan dan segera kembali. Kuingat-ingat belokan yang tadi kulewati. Aku menoleh ke kanan dan kiri, tetapi Ibu tidak terlihat. Bagaimana ini? Di mana aku? Dadaku mulai berdebar-debar. Mungkin aku salah belok, mungkin aku tersesat!

        Output:
        [
            {
                "question": "Pada pukul berapa penulis dan Ibunya sampai di stasiun tujuan?",
                "answer": "Pukul 09.00"
            },
            {
                "question": "Apa yang dilakukan penulis setelah sampai di stasiun?",
                "answer": "meminta izin ke kamar kecil."
            },
            {
                "question": "Apa yang dilakukan penulis setelah selesai buang air kecil?",
                "answer": "mencuci tangan"
            }
        ]


        Input:
        Kemarin ibu guru memberitahukan kepada kami bahwa hari ini akan ada pemberian vitamin. Kami diminta membawa surat permohonan izin orang tua untuk ditandatangani. Ibu guru mengatakan jika disuntik hanya sakit sebentar lalu kami akan kembali ceria. Hari ini tiba saatnya kami disuntik. Ani menangis ketakutan. Budi berlarian kesana kemari. Aku ingat pesan ayah bahwa aku harus menjadi anak yang pemberani. Akhirnya aku disuntik tanpa menangis. Ibu guru mengatakan bahwa aku anak yang hebat.
        
        Output:
        [
            {
                "question": "Apa yang ibu guru beritahukan kemarin?",
                "answer": "hari ini akan ada pemberian vitamin"
            },
            {
                "question": "Ibu guru meminta membawa apa?",
                "answer": "surat permohonan izin orang tua"
            },
            {
                "question": "Bagaimana reaksi Ani saat akan disuntik?",
                "answer": "Ani menangis ketakutan"
            },
            {
                "question": "Bagaimana reaksi Budi saat akan disuntik?",
                "answer": "Budi berlarian kesana kemari."
            },
            {
                "question": "Apa yang ibu guru katakan tentang penulis setelah disuntik?",
                "answer": "anak yang hebat"
            }
        ]

    '''.strip()

    VERIFY_QUESTION_ANSWER_PROMPT = '''
        Your task is to verify the wether question and answer is correct or coherent with story.

        Story: {STORY}
        Question: {QUESTION}
        Answer: {ANSWER}

        If the question and answer is correct and coherent with the story, return "True".
        Only answer with True or False.
    '''

    QUESTION_ANSWER_WIKI_PROMPT = '''
        Write multiple question and short answer about the context below delimited by the triple backtick (```) symbol:
        
        ```{CONTEXT}```

        The question and answer should be written in Bahasa Indonesia.
        The question should be suitable for student exam or test.
        The answer should be a short answer that can be found in the context.
        The answer should be extracted from the context and should be exactly the same as stated in the context, do not change the capitalization and the arrangement.
        The minimum number of question and answer is 1 and the maximum is 5.

        Your response should be in JSON format with the following keys: question, answer.

        
        Here is an example of the input and output:

        Input:
        Kesultanan Utsmaniyah, nama resmi Daulat/Negara Agung Utsmaniyah (Ottoman Turkish: دولت عليه عثمانیه Devlet-i ʿAliyye-yi ʿOmâniyye;[6] sering disebut dalam bahasa Turki modern sebagai Osmanlı İmparatorluğu (Kekaisaran Utsmaniyah) atau Osmanlı Devleti (Negara Utsmaniyah); kadang disebut Kesultanan Turki  atau Turki saja; (sering pula disebut Kekaisaran Ottoman yang diambil dari ejaan Barat) adalah kekaisaran lintas benua yang didirikan oleh suku-suku Turki di bawah pimpinan Osman Bey di barat laut Anatolia pada tahun 1299.[7] Setelah 1354, Utsmaniyah melintasi Eropa dan memulai penaklukkan Balkan, mengubah negara Utsmaniyah yang hanya berupa kadipaten kecil menjadi negara lintas benua. Utsmani mengakhiri riwayat Kekaisaran Romawi Timur seiring penaklukan Konstantinopel oleh Mehmed II tahun 1453.[8][9][10]  Sepanjang abad ke-16 dan 17, tepatnya pada puncak kekuasaannya di bawah pemerintahan Suleiman Al-Qanuni, Kesultanan Utsmaniyah adalah salah satu negara terkuat di dunia, imperium multinasional dan multibahasa yang mengendalikan sebagian besar Eropa Tenggara, Asia Barat/Kaukasus, Afrika Utara, dan Tanduk Afrika.[11]

        Output:
        [
            {
                "question": "Dimanakah Kesultanan Utsmaniyah berkuasa?",
                "answer": "barat laut Anatolia"
            },
            {
                "question": "Kapan Kesultanan Utsmaniyah didirikan?",
                "answer": "tahun 1299"
            },
            {
                "question": "Kapan Utsmani mengakhiri riwayat Kekaisaran Romawi Timur?",
                "answer": "tahun 1453"
            },
            {
                "question": "Siapa yang menaklukan Konstantinopel?",
                "answer": "Mehmed II"
            }
        ]

        
        Input:
        Penemu telepon genggam yang pertama adalah Martin Cooper, seorang karyawan Motorola pada tanggal 03 April 1973, walaupun banyak disebut-sebut penemu telepon genggam adalah sebuah tim dari salah satu divisi Motorola (divisi tempat Cooper bekerja) dengan model pertama adalah DynaTAC. Ide yang dicetuskan oleh Cooper adalah sebuah alat komunikasi yang kecil dan mudah dibawa bepergian secara fleksibel.

        Output:
        [
            {
                "question": "Siapa penemu telepon genggam pertama?",
                "answer": "Martin Cooper"
            },
            {
                "question": "Kapan telepon genggam pertama ditemukan?",
                "answer": "03 April 1973"
            },
            {
                "question": "Dimana Martin Cooper bekerja?",
                "answer": "Motorola"
            }
        ]


        Input:
        Kompleks bangunan Candi Penataran menempati areal tanah seluas 12.946 meter persegi berjajar membujur dari barat laut ke timur dan tenggara. Seluruh halaman komplek percandian kecuali yang bagian tenggara dibagi menjadi tiga bagian, yang dipisahkan oleh dua dinding. Untuk lebih mudahnya dalam memahami kompek Candi Penataran, bagian-bagian dari Candi Penataran disebut halaman depan, halaman tengah, dan halaman belakang. Susunan dari komplek Candi Penataran yang sangat unik dan tidak tersusun simetris. Hal ini mengambarkan bahwa pembuatan candi tidak dalam satu periode.
        
        Output:
        [
            {
                "question": "Berapakah luas kompleks bangunan Candi Penataran?",
                "answer": "12.946 meter persegi"
            },
            {
                "question": "Membujur dari mana ke mana kompleks bangunan Candi Penataran?",
                "answer": "dari barat laut ke timur dan tenggara"
            },
            {
                "question": "Halaman komplek percandian dibagi menjadi berapa bagian?",
                "answer": "tiga bagian"
            },
            {
                "question": "Disebut apa saja bagian-bagian dari Candi Penataran?",
                "answer": "halaman depan, halaman tengah, dan halaman belakang"
            },
            {
                "question": "Apakah susunan dari komplek Candi Penataran simetris?",
                "answer": "Tidak"
            }
        ]

    '''.strip()

    VERIFY_QUESTION_ANSWER_WIKI_PROMPT = '''
        Your task is to verify the wether question and answer is correct or coherent with context.

        Context: {CONTEXT}
        Question: {QUESTION}
        Answer: {ANSWER}

        If the question and answer is correct and coherent with the context, return "True".
        Only answer with True or False.
    '''