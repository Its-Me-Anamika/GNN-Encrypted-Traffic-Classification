import React from 'react';
import { LayoutDashboard, Network, BarChart2, Cpu, FileText, Settings } from 'lucide-react';

export const Sidebar = ({ activeTab = 'dashboard', setActiveTab }) => {
  const navItems = [
    { id: 'dashboard', label: 'Live Dashboard', icon: LayoutDashboard },
    { id: 'graph', label: 'Network Topology', icon: Network },
    { id: 'analytics', label: 'Traffic Analytics', icon: BarChart2 },
    { id: 'performance', label: 'Model Performance', icon: Cpu },
    { id: 'docs', label: 'Documentation', icon: FileText }
  ];

  return (
    <aside
      style={{
        width: '240px',
        background: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-color)',
        padding: '1.5rem 1rem',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between'
      }}
    >
      <div>
        <div style={{ padding: '0 0.8rem 1.5rem', color: 'var(--text-dim)', fontSize: '0.7rem', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Navigation
        </div>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.8rem',
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  border: 'none',
                  background: isActive ? 'rgba(0, 242, 254, 0.12)' : 'transparent',
                  color: isActive ? 'var(--primary-cyan)' : 'var(--text-muted)',
                  borderLeft: isActive ? '3px solid var(--primary-cyan)' : '3px solid transparent',
                  fontWeight: isActive ? '600' : '400',
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.2s ease'
                }}
              >
                <Icon style={{ width: '18px', height: '18px' }} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="glass-card" style={{ padding: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
        <p style={{ fontWeight: '600', color: 'var(--text-main)', marginBottom: '0.2rem' }}>GNN Model Active</p>
        <p>GraphSAGE (PyG v2.3)</p>
        <p style={{ color: 'var(--cat-file)', marginTop: '0.4rem' }}>● Monte Carlo Dropout</p>
      </div>
    </aside>
  );
};

export default Sidebar;
