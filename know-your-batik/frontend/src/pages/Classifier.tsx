import { useCallback, useRef, useState } from 'react'
import { getBatikInfo, predictBatik } from '../api/batikApi.ts'
import type { BatikInfo, PredictionResponse } from '../api/batikApi.ts'
import BatikInfoCard from '../components/BatikInfoCard.tsx'

const ALLOWED = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
const MAX_MB = 10

function formatName(name: string) {
  return name.replace(/_/g, ' ')
}

export default function Classifier() {
  const [preview, setPreview] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [batikInfo, setBatikInfo] = useState<BatikInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const reset = () => {
    setResult(null)
    setBatikInfo(null)
    setError(null)
  }

  const handleFile = useCallback((f: File) => {
    if (!ALLOWED.includes(f.type)) {
      setError('Format tidak didukung. Gunakan JPG, PNG, atau WEBP.')
      return
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`Ukuran file maksimal ${MAX_MB} MB.`)
      return
    }
    reset()
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }, [])

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
    e.target.value = ''
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const res = await predictBatik(file)
      setResult(res)
      const info = await getBatikInfo(res.predicted_class)
      setBatikInfo(info)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Terjadi kesalahan. Coba lagi.')
    } finally {
      setLoading(false)
    }
  }

  const clearAll = () => {
    setFile(null)
    setPreview(null)
    reset()
  }

  return (
    <div className="min-h-screen bg-batik-cream py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-2">
            Identifikasi Motif
          </p>
          <h1 className="font-serif text-4xl text-batik-brown font-semibold">
            Kenali Batikmu
          </h1>
          <p className="mt-3 text-batik-brown/60 max-w-lg mx-auto">
            Unggah foto batik untuk mengidentifikasi motif dan mengetahui asal-usul budayanya
          </p>
        </div>

        {/* Upload area */}
        <div
          className={`relative batik-card rounded-sm cursor-pointer transition-all duration-200 ${
            dragOver ? 'bg-batik-gold/10 scale-[1.01]' : 'bg-white hover:bg-batik-cream/50'
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !preview && inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/webp"
            className="hidden"
            onChange={handleChange}
          />

          {preview ? (
            <div className="p-6">
              <div className="flex flex-col md:flex-row gap-6 items-start">
                {/* Preview */}
                <div className="relative shrink-0">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-48 h-48 object-cover rounded-sm border-2 border-batik-gold"
                  />
                  <button
                    onClick={(e) => { e.stopPropagation(); clearAll() }}
                    className="absolute -top-3 -right-3 w-7 h-7 bg-batik-red text-white rounded-full text-xs font-bold hover:bg-red-800 transition-colors flex items-center justify-center"
                    title="Hapus"
                  >
                    ✕
                  </button>
                </div>

                {/* File info + action */}
                <div className="flex-1 flex flex-col justify-between min-h-[192px]">
                  <div>
                    <p className="text-sm font-semibold text-batik-brown">{file?.name}</p>
                    <p className="text-xs text-batik-brown/50 mt-1">
                      {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : ''}
                    </p>
                  </div>

                  <div className="flex gap-3 mt-6">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleSubmit() }}
                      disabled={loading}
                      className="btn-primary disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      {loading ? (
                        <>
                          <div className="w-4 h-4 border-2 border-batik-brown/30 border-t-batik-brown rounded-full animate-spin" />
                          Menganalisis...
                        </>
                      ) : (
                        <>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                          </svg>
                          Identifikasi
                        </>
                      )}
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
                      className="btn-outline text-sm"
                    >
                      Ganti Gambar
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-20 text-center px-6">
              <div className="mx-auto w-20 h-20 rounded-full bg-batik-gold/10 flex items-center justify-center mb-5 border-2 border-batik-gold/30 border-dashed">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#C8960C" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17,8 12,3 7,8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p className="font-serif text-lg text-batik-brown font-medium">
                Seret gambar ke sini atau klik untuk memilih
              </p>
              <p className="text-sm text-batik-brown/50 mt-2">
                JPG, PNG, WEBP — Maksimal {MAX_MB} MB
              </p>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-sm text-red-700 text-sm">
            ⚠ {error}
          </div>
        )}

        {/* Loading overlay */}
        {loading && (
          <div className="mt-8 text-center py-12">
            <div className="batik-spinner mx-auto" />
            <p className="mt-4 text-batik-brown/60 text-sm font-medium animate-pulse">
              Menganalisis motif batik...
            </p>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="mt-8 space-y-6">
            {/* Divider */}
            <div className="flex items-center gap-4">
              <div className="flex-1 h-px bg-batik-gold/30" />
              <span className="text-xs font-semibold uppercase tracking-widest text-batik-gold">
                Hasil Identifikasi
              </span>
              <div className="flex-1 h-px bg-batik-gold/30" />
            </div>

            {/* Top prediction banner */}
            <div className="batik-card p-6 bg-batik-brown text-center">
              <p className="text-xs font-semibold uppercase tracking-widest text-batik-gold/70 mb-1">
                Motif Teridentifikasi
              </p>
              <h2 className="font-serif text-3xl md:text-4xl text-batik-gold font-bold mt-1">
                {formatName(result.predicted_class)}
              </h2>
              <div className="mt-3 flex items-center justify-center gap-3">
                <div className="h-2 w-32 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-batik-gold rounded-full transition-all duration-1000"
                    style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
                  />
                </div>
                <span className="text-2xl font-bold text-batik-gold">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-batik-cream/50 text-xs mt-2">
                Diproses dalam {result.processing_time_ms.toFixed(0)} ms
              </p>
            </div>

            {/* Top 5 bar chart */}
            <div className="batik-card p-6 bg-white">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-5">
                5 Prediksi Teratas
              </h3>
              <div className="space-y-4">
                {result.top_predictions.map((p) => {
                  const pct = (p.confidence * 100).toFixed(1)
                  const isTop = p.rank === 1
                  return (
                    <div key={p.rank}>
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span
                            className={`w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center ${
                              isTop
                                ? 'bg-batik-gold text-batik-brown'
                                : 'bg-gray-100 text-gray-500'
                            }`}
                          >
                            {p.rank}
                          </span>
                          <span className={`text-sm font-medium ${isTop ? 'text-batik-brown' : 'text-gray-600'}`}>
                            {formatName(p.class_name)}
                          </span>
                        </div>
                        <span className={`text-sm font-bold tabular-nums ${isTop ? 'text-batik-gold' : 'text-gray-400'}`}>
                          {pct}%
                        </span>
                      </div>
                      <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${
                            isTop ? 'bg-batik-gold' : 'bg-batik-teal/40'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Batik info card */}
            {batikInfo && <BatikInfoCard info={batikInfo} />}
          </div>
        )}
      </div>
    </div>
  )
}
