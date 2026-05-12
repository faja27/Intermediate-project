import { useEffect, useState } from 'react'
import { getAllClasses, getBatikInfo } from '../api/batikApi.ts'
import type { BatikInfo } from '../api/batikApi.ts'
import BatikInfoCard from '../components/BatikInfoCard.tsx'

function formatName(name: string) {
  return name.replace(/_/g, ' ')
}

function getRegion(name: string): string {
  const regionMap: Record<string, string> = {
    Bali_Barong: 'Bali', Bali_Merak: 'Bali',
    Ceplok: 'Jawa Tengah', Corak_Insang: 'Kalimantan Barat',
    Ikat_Celup: 'Nasional', Jakarta_Ondel_Ondel: 'Jakarta',
    Jawa_Barat_Megamendung: 'Jawa Barat', Jawa_Timur_Pring: 'Jawa Timur',
    Kalimantan_Dayak: 'Kalimantan', Lampung_Gajah: 'Lampung',
    Lasem: 'Jawa Tengah', Madura_Mataketeran: 'Jawa Timur',
    Maluku_Pala: 'Maluku', NTB_Lumbung: 'NTB',
    Papua_Asmat: 'Papua', Papua_Cendrawasih: 'Papua', Papua_Tifa: 'Papua',
    Priangan_Merak_Ngibing: 'Jawa Barat', Sekar: 'Jawa Tengah',
    Sidoluhur: 'Jawa Tengah', Sogan: 'Jawa Tengah',
    Solo_Parang: 'Jawa Tengah', Sulawesi_Selatan_Lontara: 'Sulawesi Selatan',
    Sumatera_Barat_Rumah_Minang: 'Sumatera Barat',
    Sumatera_Utara_Boraspati: 'Sumatera Utara',
    Tambal: 'Jawa Tengah', Yogyakarta_Kawung: 'Yogyakarta',
    Yogyakarta_Parang: 'Yogyakarta',
  }
  return regionMap[name] ?? 'Indonesia'
}

export default function Gallery() {
  const [classes, setClasses] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [modalInfo, setModalInfo] = useState<BatikInfo | null>(null)
  const [modalLoading, setModalLoading] = useState(false)
  const [search, setSearch] = useState('')

  useEffect(() => {
    getAllClasses()
      .then(setClasses)
      .catch(() => setError('Gagal memuat daftar motif.'))
      .finally(() => setLoading(false))
  }, [])

  const openModal = async (cls: string) => {
    setSelected(cls)
    setModalInfo(null)
    setModalLoading(true)
    try {
      const info = await getBatikInfo(cls)
      setModalInfo(info)
    } catch {
      /* modal will show minimal info */
    } finally {
      setModalLoading(false)
    }
  }

  const filtered = classes.filter((c) =>
    formatName(c).toLowerCase().includes(search.toLowerCase()) ||
    getRegion(c).toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="min-h-screen bg-batik-cream py-12 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-2">
            Koleksi Motif
          </p>
          <h1 className="font-serif text-4xl text-batik-brown font-semibold">Galeri Batik</h1>
          <p className="mt-3 text-batik-brown/60 max-w-lg mx-auto">
            28 motif batik dari berbagai penjuru Nusantara
          </p>
        </div>

        {/* Search */}
        <div className="max-w-md mx-auto mb-8">
          <div className="relative">
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 text-batik-brown/40"
              width="18" height="18" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
            </svg>
            <input
              type="text"
              placeholder="Cari motif atau daerah..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-3 border-2 border-batik-gold/30 rounded-sm bg-white text-batik-brown placeholder-batik-brown/30 focus:outline-none focus:border-batik-gold transition-colors"
            />
          </div>
        </div>

        {loading && (
          <div className="text-center py-20">
            <div className="batik-spinner mx-auto" />
            <p className="mt-4 text-batik-brown/50 text-sm animate-pulse">Memuat motif batik...</p>
          </div>
        )}
        {error && <p className="text-center text-red-600 py-10">{error}</p>}

        {/* Grid */}
        {!loading && !error && (
          <>
            <p className="text-xs text-batik-brown/40 mb-4">{filtered.length} motif ditemukan</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {filtered.map((cls) => (
                <button
                  key={cls}
                  onClick={() => openModal(cls)}
                  className="batik-card p-5 bg-white hover:bg-batik-cream text-left transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md group"
                >
                  {/* Corner ornaments */}
                  <svg className="absolute top-1 left-1 w-3 h-3 text-batik-gold/50" viewBox="0 0 16 16" fill="none">
                    <path d="M0 0 L10 0 L10 2 L2 2 L2 10 L0 10 Z" fill="currentColor" />
                  </svg>
                  <svg className="absolute bottom-1 right-1 w-3 h-3 text-batik-gold/50 rotate-180" viewBox="0 0 16 16" fill="none">
                    <path d="M0 0 L10 0 L10 2 L2 2 L2 10 L0 10 Z" fill="currentColor" />
                  </svg>

                  <div className="w-8 h-8 mx-auto mb-3 rounded-full bg-batik-gold/10 flex items-center justify-center group-hover:bg-batik-gold/20 transition-colors">
                    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                      <circle cx="10" cy="10" r="8" stroke="#C8960C" strokeWidth="1" />
                      <circle cx="10" cy="10" r="3" fill="#C8960C" />
                    </svg>
                  </div>

                  <h3 className="font-serif text-batik-brown text-sm font-semibold leading-snug text-center">
                    {formatName(cls)}
                  </h3>

                  <div className="mt-2 text-center">
                    <span className="text-[10px] bg-batik-teal/10 text-batik-teal px-2 py-0.5 rounded-full font-medium">
                      {getRegion(cls)}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Modal */}
      {selected && (
        <div
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="bg-batik-cream max-w-lg w-full max-h-[90vh] overflow-y-auto rounded-sm shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="bg-batik-brown px-6 py-4 flex items-center justify-between">
              <h2 className="font-serif text-batik-gold font-semibold">
                {formatName(selected)}
              </h2>
              <button
                onClick={() => setSelected(null)}
                className="text-batik-cream/60 hover:text-batik-gold transition-colors text-xl leading-none"
              >
                ✕
              </button>
            </div>

            <div className="p-6">
              {modalLoading ? (
                <div className="py-12 text-center">
                  <div className="batik-spinner mx-auto" />
                  <p className="mt-3 text-batik-brown/50 text-sm animate-pulse">Memuat informasi...</p>
                </div>
              ) : modalInfo ? (
                <BatikInfoCard info={modalInfo} />
              ) : (
                <p className="text-batik-brown/60 text-sm">Informasi tidak tersedia.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
