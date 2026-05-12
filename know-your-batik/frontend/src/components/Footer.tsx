export default function Footer() {
  return (
    <footer className="bg-batik-brown text-batik-cream">
      {/* Batik stripe */}
      <div
        className="h-1 w-full"
        style={{
          background:
            'repeating-linear-gradient(90deg, #C8960C 0px, #C8960C 12px, #8B1A1A 12px, #8B1A1A 24px, #1A4A4A 24px, #1A4A4A 36px, #C8960C 36px)',
        }}
      />

      <div className="max-w-6xl mx-auto px-4 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <h3 className="font-serif text-batik-gold text-lg mb-2">Know Your Batik</h3>
            <p className="text-sm text-batik-cream/70 leading-relaxed">
              Mengenali dan melestarikan warisan budaya batik Indonesia melalui kecerdasan buatan.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-3">
              Navigasi
            </h4>
            <ul className="space-y-2 text-sm text-batik-cream/70">
              {['Beranda', 'Identifikasi', 'Galeri', 'Pelajari'].map((l) => (
                <li key={l}>
                  <a
                    href={`/${l === 'Beranda' ? '' : l.toLowerCase()}`}
                    className="hover:text-batik-gold transition-colors"
                  >
                    {l}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Attribution */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-3">
              Tentang
            </h4>
            <p className="text-sm text-batik-cream/70 leading-relaxed">
              Model ResNet-50 yang dilatih dengan 2.128 gambar batik dari 28 kelas motif nusantara.
            </p>
            <p className="text-xs text-batik-cream/40 mt-4">
              © {new Date().getFullYear()} Know Your Batik
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}
