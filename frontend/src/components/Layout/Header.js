import React from 'react';
import { Activity, ShieldCheck, Wifi, Radio } from 'lucide-react';

export const Header = ({ wsStatus = 'connected' }) => {
  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '1rem 2rem',
        background: 'rgba(11, 15, 25, 0.9)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid var(--border-color)',
        sticky: 'top'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, var(--primary-cyan), var(--primary-blue))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#000'
          }}
        >
          <Activity style={{ width: '22px', height: '22px' }} />
        </div>
        <div>
          <h1 style={{ fontSize: '1.2rem', fontWeight: '700', letterSpacing: '-0.02em' }}>
            <span className="gradient-text">Encrypted Traffic GNN</span> Classifier
          </h1>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Fine-Grained Flow Identification via Graph Neural Networks (GraphSAGE / GAT)
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <ShieldCheck style={{ width: '16px', height: '16px', color: 'var(--cat-file)' }} />
          <span>ISCX Dataset Engine</span>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.4rem 0.8rem',
            borderRadius: '20px',
            background: wsStatus === 'connected' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
            border: `1px solid ${wsStatus === 'connected' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            fontSize: '0.75rem',
            fontWeight: '600',
            color: wsStatus === 'connected' ? '#10b981' : '#ef4444'
          }}
        >
          <Radio style={{ width: '14px', height: '14px', animation: wsStatus === 'connected' ? 'pulse 2s infinite' : 'none' }} />
          <span>{wsStatus === 'connected' ? 'LIVE STREAM' : 'OFFLINE'}</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
