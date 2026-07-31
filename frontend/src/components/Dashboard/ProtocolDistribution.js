import React from 'react';

export const ProtocolDistribution = ({ protocols = { TCP: 0, UDP: 0 } }) => {
  const total = (protocols.TCP || 0) + (protocols.UDP || 0) || 1;
  const tcpPct = (((protocols.TCP || 0) / total) * 100).toFixed(1);
  const udpPct = (((protocols.UDP || 0) / total) * 100).toFixed(1);

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <h3 style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: '600' }}>
        Transport Protocol Split
      </h3>

      <div style={{ display: 'flex', gap: '1rem', marginTop: '0.4rem' }}>
        <div style={{ flex: 1, background: 'rgba(0, 242, 254, 0.08)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(0, 242, 254, 0.2)' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TCP Encrypted</span>
          <h3 className="mono" style={{ fontSize: '1.4rem', color: 'var(--primary-cyan)', marginTop: '0.2rem' }}>{tcpPct}%</h3>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{protocols.TCP || 0} flows</span>
        </div>

        <div style={{ flex: 1, background: 'rgba(127, 0, 255, 0.08)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(127, 0, 255, 0.2)' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>UDP / QUIC</span>
          <h3 className="mono" style={{ fontSize: '1.4rem', color: 'var(--cat-streaming)', marginTop: '0.2rem' }}>{udpPct}%</h3>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{protocols.UDP || 0} flows</span>
        </div>
      </div>
    </div>
  );
};

export default ProtocolDistribution;
