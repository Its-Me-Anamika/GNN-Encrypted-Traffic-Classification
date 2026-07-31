export const formatBytes = (bytes, decimals = 2) => {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

export const formatDuration = (seconds) => {
  if (!seconds || seconds <= 0) return '0s';
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(0);
  return `${mins}m ${secs}s`;
};

export const getCategoryBadgeClass = (category) => {
  if (!category) return 'badge-unknown';
  const normalized = category.toLowerCase().replace(/\s+/g, '-');
  switch (normalized) {
    case 'streaming': return 'badge-streaming';
    case 'chat': return 'badge-chat';
    case 'voip': return 'badge-voip';
    case 'file-transfer': return 'badge-file-transfer';
    case 'browsing': return 'badge-browsing';
    case 'p2p': return 'badge-p2p';
    case 'email': return 'badge-email';
    case 'vpn': return 'badge-vpn';
    default: return 'badge-unknown';
  }
};

export const getCategoryColor = (category) => {
  const map = {
    'Streaming': '#a855f7',
    'Chat': '#3b82f6',
    'VoIP': '#06b6d4',
    'File Transfer': '#10b981',
    'Browsing': '#f59e0b',
    'P2P': '#ef4444',
    'Email': '#ec4899',
    'VPN': '#8b5cf6',
    'Unknown': '#64748b'
  };
  return map[category] || '#64748b';
};
