import { Link } from 'react-router-dom'

const FEATURED = [
  'Bali_Barong',
  'Jawa_Barat_Megamendung',
  'Yogyakarta_Kawung',
  'Papua_Cendrawasih',
  'Kalimantan_Dayak',
  'Lasem',
]

function formatName(name: string) {
  return name.replace(/_/g, ' ')
}

const REGION_MAP: Record<string, string> = {
  Bali_Barong: 'Bali',
  Jawa_Barat_Megamendung: 'Cirebon, Jawa Barat',
  Yogyakarta_Kawung: 'Yogyakarta',
  Papua_Cendrawasih: 'Papua',
  Kalimantan_Dayak: 'Kalimantan',
  Lasem: 'Rembang, Jawa Tengah',
}

const STATS = [
  { value: '28', label: 'Motif Batik' },
  { value: '2.128', label: 'Gambar Dataset' },
  { value: '85,9%', label: 'Akurasi Model' },
]

export default function Home() {
  return (
    <div>
      {/* ── Hero ── */}
      <section className="relative min-h-[92vh] flex items-center justify-center bg-batik-brown overflow-hidden">
        {/* Kawung texture overlay */}
        <div
          className="absolute inset-0 bg-batik-pattern opacity-30"
          style={{ backgroundSize: '80px 80px' }}
        />
        {/* Gradient vignette */}
        <div className="absolute inset-0 bg-gradient-to-b from-batik-brown/60 via-transparent to-batik-brown/80" />

        <div className="relative z-10 text-center px-6 max-w-3xl mx-auto">
          {/* Decorative ornament */}
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="h-px w-20 bg-batik-gold/60" />
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="14" stroke="#C8960C" strokeWidth="1.5" />
              <circle cx="16" cy="16" r="8" stroke="#C8960C" strokeWidth="1" />
              <circle cx="16" cy="16" r="3" fill="#C8960C" />
              <circle cx="16" cy="2" r="2" fill="#C8960C" />
              <circle cx="16" cy="30" r="2" fill="#C8960C" />
              <circle cx="2" cy="16" r="2" fill="#C8960C" />
              <circle cx="30" cy="16" r="2" fill="#C8960C" />
            </svg>
            <div className="h-px w-20 bg-batik-gold/60" />
          </div>

          <h1 className="font-serif text-5xl md:text-7xl text-batik-gold font-bold leading-tight drop-shadow-lg">
            Know Your Batik
          </h1>
          <p className="mt-5 text-lg md:text-xl text-batik-cream/90 leading-relaxed max-w-xl mx-auto">
            Kenali warisan budaya batik Indonesia melalui kecerdasan buatan. Unggah gambar dan temukan motif batiknya.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/classifier" className="btn-primary text-base">
              Mulai Identifikasi
            </Link>
            <Link
              to="/gallery"
              className="inline-block border-2 border-batik-gold text-batik-gold px-8 py-3 font-bold rounded-sm hover:bg-batik-gold hover:text-batik-brown transition-all duration-200"
            >
              Lihat Galeri
            </Link>
          </div>
        </div>

        {/* Bottom wave */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 60" preserveAspectRatio="none" className="w-full h-12 md:h-16">
            <path
              d="M0 30 Q180 0 360 30 Q540 60 720 30 Q900 0 1080 30 Q1260 60 1440 30 L1440 60 L0 60 Z"
              fill="#FAF3E0"
            />
          </svg>
        </div>
      </section>

      {/* ── Stats ── */}
      <section className="bg-batik-cream py-16">
        <div className="max-w-4xl mx-auto px-6">
          <div className="grid grid-cols-3 gap-4 md:gap-12 text-center">
            {STATS.map(({ value, label }) => (
              <div key={label} className="group">
                <div className="font-serif text-4xl md:text-6xl text-batik-gold font-bold group-hover:scale-110 transition-transform duration-300">
                  {value}
                </div>
                <div className="mt-2 text-sm md:text-base text-batik-brown/70 font-medium uppercase tracking-wider">
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Ornament divider */}
        <div className="flex items-center gap-4 max-w-4xl mx-auto px-6 mt-12">
          <div className="flex-1 h-px bg-batik-gold/30" />
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <rect x="2" y="2" width="20" height="20" stroke="#C8960C" strokeWidth="1" transform="rotate(45 12 12)" />
            <circle cx="12" cy="12" r="3" fill="#C8960C" />
          </svg>
          <div className="flex-1 h-px bg-batik-gold/30" />
        </div>
      </section>

      {/* ── Featured Motifs ── */}
      <section className="bg-batik-brown py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <p className="text-xs font-semibold uppercase tracking-widest text-batik-gold/70 mb-2">
              Koleksi Pilihan
            </p>
            <h2 className="font-serif text-3xl md:text-4xl text-batik-gold">
              Motif Batik Nusantara
            </h2>
            <p className="mt-3 text-batik-cream/70 max-w-lg mx-auto">
              Jelajahi kekayaan motif batik dari berbagai penjuru Indonesia
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 md:gap-6">
            {FEATURED.map((cls) => (
              <Link key={cls} to="/gallery" className="group block">
                <div className="batik-card p-6 bg-batik-brown/60 hover:bg-batik-teal transition-colors duration-300">
                  {/* Corner ornaments */}
                  <svg
                    className="absolute top-1 left-1 w-4 h-4 text-batik-gold opacity-60"
                    viewBox="0 0 16 16" fill="none"
                  >
                    <path d="M0 0 L10 0 L10 2 L2 2 L2 10 L0 10 Z" fill="currentColor" />
                  </svg>
                  <svg
                    className="absolute top-1 right-1 w-4 h-4 text-batik-gold opacity-60 rotate-90"
                    viewBox="0 0 16 16" fill="none"
                  >
                    <path d="M0 0 L10 0 L10 2 L2 2 L2 10 L0 10 Z" fill="currentColor" />
                  </svg>
                  <svg
                    className="absolute bottom-1 left-1 w-4 h-4 text-batik-gold opacity-60 -rotate-90"
                    viewBox="0 0 16 16" fill="none"
                  >
                    <path d="M0 0 L10 0 L10 2 L2 2 L2 10 L0 10 Z" fill="currentColor" />
                  </svg>
                  <svg
                    className="absolute bottom-1 right-1 w-4 h-4 text-batik-gold opacity-60 rotate-180"
                    viewBox="0 0 16 16" fill="none"
                  >
                    <path d="M0 0 L10 0 L10 2 L2 2 L2 10 L0 10 Z" fill="currentColor" />
                  </svg>

                  <div className="text-center py-2">
                    <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-batik-gold/20 flex items-center justify-center">
                      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <circle cx="10" cy="10" r="8" stroke="#C8960C" strokeWidth="1" />
                        <circle cx="10" cy="10" r="4" stroke="#C8960C" strokeWidth="0.8" />
                        <circle cx="10" cy="10" r="2" fill="#C8960C" />
                      </svg>
                    </div>
                    <h3 className="font-serif text-batik-gold font-semibold text-sm md:text-base leading-snug">
                      {formatName(cls)}
                    </h3>
                    <p className="mt-1 text-xs text-batik-cream/60">{REGION_MAP[cls]}</p>
                  </div>
                </div>
              </Link>
            ))}
          </div>

          <div className="text-center mt-10">
            <Link to="/gallery" className="btn-outline text-sm">
              Lihat Semua 28 Motif →
            </Link>
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="bg-batik-cream py-20">
        <div className="max-w-2xl mx-auto px-6 text-center">
          <h2 className="font-serif text-3xl md:text-4xl text-batik-brown font-semibold">
            Coba Identifikasi Batikmu
          </h2>
          <p className="mt-4 text-batik-brown/70 leading-relaxed">
            Unggah foto batik dan biarkan AI kami mengenali motif serta asal-usul budayanya dalam hitungan detik.
          </p>
          <Link to="/classifier" className="btn-primary mt-8 inline-block">
            Mulai Sekarang
          </Link>
        </div>
      </section>
    </div>
  )
}
