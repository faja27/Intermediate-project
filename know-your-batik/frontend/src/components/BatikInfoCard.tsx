import type { BatikInfo } from '../api/batikApi.ts'

interface Props {
  info: BatikInfo
}

function formatName(name: string) {
  return name.replace(/_/g, ' ')
}

export default function BatikInfoCard({ info }: Props) {
  return (
    <div className="batik-card p-6 bg-white">
      {/* Header */}
      <div className="mb-4 pb-4 border-b border-batik-gold/30">
        <h3 className="font-serif text-2xl text-batik-brown font-semibold">
          {formatName(info.class_name)}
        </h3>
        <div className="flex flex-wrap gap-2 mt-2">
          <span className="inline-flex items-center gap-1 text-xs font-medium bg-batik-brown text-batik-cream px-3 py-1 rounded-full">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" />
              <circle cx="12" cy="9" r="2.5" />
            </svg>
            {info.origin}
          </span>
          <span className="inline-flex items-center gap-1 text-xs font-medium bg-batik-teal text-batik-cream px-3 py-1 rounded-full">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <path d="M3 9h18M9 21V9" />
            </svg>
            {info.region}
          </span>
        </div>
      </div>

      {/* Description */}
      <p className="text-sm text-gray-700 leading-relaxed mb-5">{info.description}</p>

      {/* Characteristics */}
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-widest text-batik-gold mb-3">
          Karakteristik
        </h4>
        <div className="flex flex-wrap gap-2">
          {info.characteristics.map((c, i) => (
            <span
              key={i}
              className="text-xs bg-batik-cream border border-batik-gold/40 text-batik-brown px-3 py-1 rounded-full"
            >
              {c}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
