import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'

const NAV_LINKS = [
  { to: '/', label: 'Beranda' },
  { to: '/classifier', label: 'Identifikasi' },
  { to: '/gallery', label: 'Galeri' },
  { to: '/learn', label: 'Pelajari' },
]

export default function Navbar() {
  const [open, setOpen] = useState(false)

  return (
    <header className="relative z-50">
      {/* Batik pattern top border */}
      <div
        className="h-2 w-full"
        style={{
          background: 'repeating-linear-gradient(90deg, #C8960C 0px, #C8960C 12px, #8B1A1A 12px, #8B1A1A 24px, #1A4A4A 24px, #1A4A4A 36px, #C8960C 36px)',
        }}
      />

      <nav className="bg-batik-brown shadow-lg">
        <div className="max-w-6xl mx-auto px-4 flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none" className="shrink-0">
              <circle cx="18" cy="18" r="17" stroke="#C8960C" strokeWidth="1.5" />
              <ellipse cx="18" cy="18" rx="14" ry="9" stroke="#C8960C" strokeWidth="1" />
              <ellipse cx="18" cy="18" rx="9" ry="14" stroke="#C8960C" strokeWidth="1" />
              <circle cx="18" cy="18" r="4" fill="#C8960C" />
              <circle cx="18" cy="4" r="2" fill="#C8960C" />
              <circle cx="18" cy="32" r="2" fill="#C8960C" />
              <circle cx="4" cy="18" r="2" fill="#C8960C" />
              <circle cx="32" cy="18" r="2" fill="#C8960C" />
            </svg>
            <span className="font-serif text-xl text-batik-gold tracking-wide group-hover:text-yellow-300 transition-colors">
              Know Your Batik
            </span>
          </Link>

          {/* Desktop nav */}
          <ul className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map(({ to, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    `px-4 py-2 rounded-sm text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-batik-gold text-batik-brown'
                        : 'text-batik-cream hover:text-batik-gold hover:bg-white/10'
                    }`
                  }
                >
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>

          {/* Mobile hamburger */}
          <button
            className="md:hidden text-batik-cream p-2"
            onClick={() => setOpen((o) => !o)}
            aria-label="Toggle menu"
          >
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
              {open ? (
                <path d="M6 6l12 12M6 18L18 6" />
              ) : (
                <path d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile menu */}
        {open && (
          <div className="md:hidden border-t border-white/10 bg-batik-brown pb-3">
            {NAV_LINKS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `block px-6 py-3 text-sm font-medium transition-colors ${
                    isActive ? 'text-batik-gold bg-white/10' : 'text-batik-cream hover:text-batik-gold'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </div>
        )}
      </nav>
    </header>
  )
}
