export function IconPlus(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="16" height="16" {...props}>
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function IconSend(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="16" height="16" {...props}>
      <path
        d="M17 3 3 9.2c-.6.27-.55 1.15.08 1.35l5.3 1.68 1.68 5.3c.2.63 1.08.68 1.35.08L17.6 3.9c.2-.44-.26-.9-.7-.7L17 3Z"
        fill="currentColor"
      />
    </svg>
  );
}

export function IconChat(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="15" height="15" {...props}>
      <path
        d="M3 4.5A1.5 1.5 0 0 1 4.5 3h11A1.5 1.5 0 0 1 17 4.5v7A1.5 1.5 0 0 1 15.5 13H8l-3.5 3v-3H4.5A1.5 1.5 0 0 1 3 11.5v-7Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconExternal(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="12" height="12" {...props}>
      <path
        d="M8 4h8v8M16 4 8.5 11.5M6 4H4.5A1.5 1.5 0 0 0 3 5.5v10A1.5 1.5 0 0 0 4.5 17h10a1.5 1.5 0 0 0 1.5-1.5V14"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconDot({ className = "", ...props }) {
  return (
    <svg viewBox="0 0 8 8" width="8" height="8" className={className} {...props}>
      <circle cx="4" cy="4" r="4" fill="currentColor" />
    </svg>
  );
}

export function IconMore(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="15" height="15" {...props}>
      <circle cx="10" cy="4.5" r="1.3" fill="currentColor" />
      <circle cx="10" cy="10" r="1.3" fill="currentColor" />
      <circle cx="10" cy="15.5" r="1.3" fill="currentColor" />
    </svg>
  );
}

export function IconEdit(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="14" height="14" {...props}>
      <path
        d="M13.5 3.5 16.5 6.5 7 16H4v-3l9.5-9.5Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconTrash(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="14" height="14" {...props}>
      <path
        d="M4 6h12M8 6V4.5A1.5 1.5 0 0 1 9.5 3h1A1.5 1.5 0 0 1 12 4.5V6m-6.5 0 .6 9.4A1.5 1.5 0 0 0 7.6 17h4.8a1.5 1.5 0 0 0 1.5-1.6L14.5 6"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconCheck(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="14" height="14" {...props}>
      <path d="M4 10.5 8 14.5 16 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconLoader({ className = "", ...props }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="16" height="16" className={`animate-spin ${className}`} {...props}>
      <circle cx="10" cy="10" r="7.5" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2" />
      <path d="M17.5 10a7.5 7.5 0 0 0-7.5-7.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function IconSparkle(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="14" height="14" {...props}>
      <path
        d="M10 3l1.2 4.3L15.5 8.5l-4.3 1.2L10 14l-1.2-4.3L4.5 8.5l4.3-1.2L10 3Z"
        fill="currentColor"
      />
    </svg>
  );
}

export function IconShield(props) {
  return (
    <svg viewBox="0 0 20 20" fill="none" width="14" height="14" {...props}>
      <path
        d="M10 2.5 16 5v5c0 4-2.6 6.7-6 7.5-3.4-.8-6-3.5-6-7.5V5l6-2.5Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path d="M7.3 10 9.3 12l3.4-4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}