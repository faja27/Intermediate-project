const SECTIONS = [
  {
    id: 'apa-itu-batik',
    title: 'Apa itu Batik?',
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <circle cx="14" cy="14" r="12" stroke="#C8960C" strokeWidth="1.5" />
        <circle cx="14" cy="14" r="7" stroke="#C8960C" strokeWidth="1" />
        <circle cx="14" cy="14" r="3" fill="#C8960C" />
      </svg>
    ),
    content: `Batik adalah kain bergambar yang pembuatannya secara khusus dengan menuliskan atau menerakan malam pada kain, kemudian pengolahannya melalui proses tertentu. Kata "batik" berasal dari bahasa Jawa "amba" yang berarti lebar, luas, dan "tik" yang berarti titik.

Batik merupakan warisan budaya yang telah ada di Indonesia sejak ratusan tahun lalu. Awalnya batik hanya dikerjakan dalam lingkungan keraton dan dipakai oleh kalangan terbatas. Namun, seiring perkembangan zaman, batik telah menyebar ke seluruh pelosok nusantara dan menjadi milik semua kalangan masyarakat.`,
  },
  {
    id: 'sejarah',
    title: 'Sejarah Batik Indonesia',
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <rect x="4" y="4" width="20" height="20" stroke="#C8960C" strokeWidth="1.5" rx="1" />
        <line x1="4" y1="10" x2="24" y2="10" stroke="#C8960C" strokeWidth="1" />
        <line x1="10" y1="4" x2="10" y2="24" stroke="#C8960C" strokeWidth="1" />
      </svg>
    ),
    content: `Sejarah batik di Indonesia berkaitan erat dengan perkembangan kerajaan Majapahit dan penyebaran ajaran Islam di Pulau Jawa. Dalam beberapa catatan, pengembangan batik banyak dilakukan pada masa-masa kerajaan Mataram, kemudian pada masa kerajaan Solo dan Yogyakarta.

Pada abad ke-17 dan 18, batik mulai dikenal lebih luas ketika para pedagang dan pelancong asing datang ke Nusantara. Pada masa penjajahan Belanda, batik mulai diekspor ke pasar Eropa dan mendapatkan apresiasi tinggi. Setelah kemerdekaan Indonesia, batik semakin dikembangkan sebagai identitas nasional dan simbol kebanggaan bangsa.`,
  },
  {
    id: 'unesco',
    title: 'Pengakuan UNESCO',
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <polygon points="14,3 17.5,10.5 26,11.5 20,17.5 21.5,26 14,22 6.5,26 8,17.5 2,11.5 10.5,10.5" stroke="#C8960C" strokeWidth="1.5" fill="none" />
        <circle cx="14" cy="14" r="3" fill="#C8960C" />
      </svg>
    ),
    content: `Pada tanggal 2 Oktober 2009, UNESCO resmi mengakui batik Indonesia sebagai Warisan Budaya Tak Benda Kemanusiaan (Intangible Cultural Heritage of Humanity). Pengakuan ini merupakan pencapaian bersejarah yang mempertegas posisi batik sebagai identitas budaya bangsa Indonesia di mata dunia.

Setiap tanggal 2 Oktober diperingati sebagai Hari Batik Nasional, di mana seluruh masyarakat Indonesia dianjurkan untuk mengenakan batik sebagai wujud kecintaan dan kebanggaan terhadap warisan leluhur. Pengakuan UNESCO ini juga mendorong pemerintah dan masyarakat untuk lebih serius dalam pelestarian dan pengembangan batik Indonesia.`,
  },
  {
    id: 'teknik',
    title: 'Teknik Pembuatan Batik',
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M6 22 L22 6" stroke="#C8960C" strokeWidth="2" strokeLinecap="round" />
        <circle cx="8" cy="20" r="4" stroke="#C8960C" strokeWidth="1.5" />
        <path d="M12 16 L22 6 L24 8" stroke="#C8960C" strokeWidth="1.5" />
      </svg>
    ),
    content: `Ada tiga teknik utama pembuatan batik yang dikenal di Indonesia:

**Batik Tulis** — Dibuat dengan menggunakan canting, alat tradisional berupa tabung kecil berisi malam (lilin batik) cair. Proses pembuatannya memerlukan keahlian, kesabaran, dan ketelitian tinggi karena dikerjakan secara manual. Satu lembar batik tulis berkualitas tinggi bisa memerlukan waktu berbulan-bulan untuk diselesaikan.

**Batik Cap** — Menggunakan cap (stempel) tembaga yang sudah memiliki motif. Proses pembuatannya lebih cepat dari batik tulis, namun tetap memerlukan keahlian khusus. Motif yang dihasilkan lebih seragam dan rapi.

**Batik Printing (Sablon)** — Metode modern menggunakan mesin untuk mencetak motif batik di atas kain. Proses ini paling cepat dan dapat diproduksi dalam jumlah besar, namun nilai seni dan budayanya dianggap lebih rendah dibanding batik tulis atau cap.`,
  },
  {
    id: 'cara-membaca',
    title: 'Cara Membaca Motif Batik',
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M4 6h20M4 12h20M4 18h14" stroke="#C8960C" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="22" cy="20" r="4" stroke="#C8960C" strokeWidth="1.5" />
        <path d="M25 23 L27 25" stroke="#C8960C" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    content: `Membaca motif batik memerlukan pemahaman tentang simbol dan makna yang terkandung dalam setiap motif. Beberapa panduan dasar:

**Identifikasi Asal Daerah** — Setiap daerah di Indonesia memiliki ciri khas motif dan warna tersendiri. Batik pesisir (Cirebon, Lasem, Pekalongan) umumnya berwarna lebih cerah dan memiliki pengaruh budaya luar. Batik pedalaman (Solo, Yogyakarta) cenderung menggunakan warna sogan (coklat) dengan motif yang lebih konservatif.

**Perhatikan Pola Utama** — Setiap batik memiliki pola utama (motif) dan pola pengisi (isian). Motif utama biasanya mencerminkan nilai filosofis atau budaya setempat, seperti motif parang yang melambangkan kekuatan, atau motif kawung yang melambangkan kesucian.

**Warna dan Maknanya** — Warna dalam batik bukan sekadar estetika, tapi mengandung makna. Merah melambangkan keberanian, hitam melambangkan keabadian, putih melambangkan kesucian, dan warna sogan (coklat) melambangkan kesederhanaan dan kebijaksanaan.

**Konteks Penggunaan** — Beberapa motif batik hanya dipakai pada acara tertentu. Motif parang rusak, misalnya, di era kerajaan hanya boleh dipakai oleh raja dan keluarga kerajaan.`,
  },
]

function SectionDivider() {
  return (
    <div className="flex items-center gap-4 my-8">
      <div className="flex-1 h-px bg-batik-gold/25" />
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
        <rect x="1" y="1" width="18" height="18" stroke="#C8960C" strokeWidth="1" transform="rotate(45 10 10)" />
        <circle cx="10" cy="10" r="3" fill="#C8960C" />
      </svg>
      <div className="flex-1 h-px bg-batik-gold/25" />
    </div>
  )
}

export default function Learn() {
  return (
    <div className="min-h-screen bg-batik-cream py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <p className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-2">
            Edukasi Budaya
          </p>
          <h1 className="font-serif text-4xl md:text-5xl text-batik-brown font-semibold">
            Mengenal Batik
          </h1>
          <p className="mt-4 text-batik-brown/60 max-w-lg mx-auto leading-relaxed">
            Pelajari sejarah, teknik pembuatan, dan makna di balik warisan budaya batik Indonesia
          </p>

          {/* UNESCO badge */}
          <div className="mt-6 inline-flex items-center gap-2 bg-batik-brown text-batik-cream px-4 py-2 rounded-full text-xs font-medium">
            <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
              <polygon points="14,2 17,9.5 25,10.5 19,16.5 20.5,25 14,21 7.5,25 9,16.5 3,10.5 11,9.5" fill="#C8960C" />
            </svg>
            Warisan Budaya Tak Benda UNESCO sejak 2009
          </div>
        </div>

        {/* TOC */}
        <nav className="batik-card p-5 bg-white mb-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-3">
            Daftar Isi
          </p>
          <ul className="space-y-2">
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  className="flex items-center gap-2 text-sm text-batik-brown hover:text-batik-gold transition-colors group"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-batik-gold/40 group-hover:bg-batik-gold transition-colors shrink-0" />
                  {s.title}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        {/* Sections */}
        {SECTIONS.map((section, idx) => (
          <div key={section.id}>
            <article id={section.id} className="batik-card p-7 bg-white">
              <div className="flex items-start gap-4">
                <div className="shrink-0 w-12 h-12 rounded-full bg-batik-gold/10 flex items-center justify-center">
                  {section.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <h2 className="font-serif text-2xl text-batik-brown font-semibold mb-4">
                    {section.title}
                  </h2>
                  <div className="prose prose-sm max-w-none">
                    {section.content.split('\n\n').map((para, i) => {
                      if (para.startsWith('**') && para.includes('**')) {
                        const parts = para.split(/\*\*(.+?)\*\*/)
                        return (
                          <p key={i} className="text-batik-brown/80 leading-relaxed mb-3">
                            {parts.map((part, j) =>
                              j % 2 === 1 ? (
                                <strong key={j} className="text-batik-brown font-semibold">
                                  {part}
                                </strong>
                              ) : (
                                part
                              )
                            )}
                          </p>
                        )
                      }
                      return (
                        <p key={i} className="text-batik-brown/80 leading-relaxed mb-3">
                          {para}
                        </p>
                      )
                    })}
                  </div>
                </div>
              </div>
            </article>
            {idx < SECTIONS.length - 1 && <SectionDivider />}
          </div>
        ))}

        {/* Call to action */}
        <div className="mt-12 bg-batik-brown batik-card text-center p-10">
          <h3 className="font-serif text-2xl text-batik-gold font-semibold">
            Sudah Siap Mencoba?
          </h3>
          <p className="mt-3 text-batik-cream/70 text-sm max-w-sm mx-auto">
            Gunakan AI kami untuk mengidentifikasi motif batik dari foto yang kamu miliki.
          </p>
          <a href="/classifier" className="btn-primary mt-6 inline-block">
            Coba Identifikasi Sekarang
          </a>
        </div>
      </div>
    </div>
  )
}
