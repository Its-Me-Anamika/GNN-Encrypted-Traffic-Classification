import React from 'react';
import { getCategoryColor } from '../../utils/helpers';

export const TopApplications = ({ distribution = {} }) => {
  const sorted = Object.entries(distribution)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const total = Object.values(distribution).reduce((acc, curr) => acc + curr, 0) || 1;

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <h3 style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: '600' }}>
        Top Encrypted Traffic Applications
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {sorted.map(([category, count]) => {
          const percentage = ((count / total) * 100).toFixed(1);
          const color = getCategoryColor(category);
          return (
            <div key={category}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                <span style={{ color: 'var(--text-main)', fontWeight: '500' }}>{category}</span>
                <span className="mono" style={{ color: 'var(--text-muted)' }}>{count} flows ({percentage}%)</span>
              </div>
              <div style={{ height: '6px', width: '100%', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${percentage}%`,
                    height: '100%',
                    background: color,
                    borderRadius: '3px',
                    transition: 'width 0.5s ease'
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TopApplications;
