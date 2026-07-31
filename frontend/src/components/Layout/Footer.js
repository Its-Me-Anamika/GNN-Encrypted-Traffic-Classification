import React from 'react';

export const Footer = () => {
  return (
    <footer
      style={{
        padding: '1rem 2rem',
        borderTop: '1px solid var(--border-color)',
        background: 'var(--bg-main)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '0.75rem',
        color: 'var(--text-dim)'
      }}
    >
      <div>
        <span>Fine-Grained Encrypted Traffic Classification using GNNs</span>
      </div>
      <div>
        <span>Final Year Engineering Project &bull; PyTorch Geometric &bull; FastAPI &bull; React</span>
      </div>
    </footer>
  );
};

export default Footer;
