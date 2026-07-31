import React from 'react';
import { Network, ShieldAlert, Cpu, Activity, CheckCircle, HelpCircle } from 'lucide-react';

export const MetricsCards = ({ metrics }) => {
  const cards = [
    {
      title: 'Total Flows Captured',
      value: metrics?.total_flows || 0,
      subtext: `${metrics?.classified_today || 0} classified today`,
      icon: Activity,
      color: 'var(--primary-cyan)'
    },
    {
      title: 'VPN Encapsulated Ratio',
      value: `${((metrics?.vpn_ratio || 0) * 100).toFixed(1)}%`,
      subtext: 'ISCX VPN-nonVPN pipeline',
      icon: ShieldAlert,
      color: 'var(--cat-vpn)'
    },
    {
      title: 'GNN Model Confidence',
      value: `${((metrics?.avg_confidence || 0) * 100).toFixed(1)}%`,
      subtext: 'Monte Carlo Dropout score',
      icon: CheckCircle,
      color: 'var(--cat-file)'
    },
    {
      title: 'Unknown Traffic Ratio',
      value: `${((metrics?.unknown_ratio || 0) * 100).toFixed(1)}%`,
      subtext: 'Confidence threshold < 0.65',
      icon: HelpCircle,
      color: 'var(--cat-p2p)'
    }
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem', width: '100%' }}>
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div key={idx} className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.8rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: '500' }}>{card.title}</span>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '8px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: card.color
                }}
              >
                <Icon style={{ width: '18px', height: '18px' }} />
              </div>
            </div>
            <div>
              <h2 style={{ fontSize: '1.8rem', fontWeight: '700', color: 'var(--text-main)', letterSpacing: '-0.03em' }}>
                {card.value}
              </h2>
              <p style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>{card.subtext}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default MetricsCards;
