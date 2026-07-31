import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export const TrafficTimeline = ({ hourlyData = [] }) => {
  const labels = hourlyData.length > 0
    ? hourlyData.map((d) => d.time)
    : ['10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00'];

  const counts = hourlyData.length > 0
    ? hourlyData.map((d) => d.count)
    : [12, 25, 42, 38, 55, 48, 62];

  const data = {
    labels,
    datasets: [
      {
        label: 'Flows Classified / Hour',
        data: counts,
        borderColor: '#00f2fe',
        backgroundColor: 'rgba(0, 242, 254, 0.08)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: '#00f2fe'
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#64748b', font: { size: 10 } }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#64748b', font: { size: 10 } }
      }
    }
  };

  return (
    <div className="glass-card" style={{ height: '340px', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ fontSize: '0.95rem', color: 'var(--text-main)', marginBottom: '1rem', fontWeight: '600' }}>
        Live Traffic Ingestion Timeline
      </h3>
      <div style={{ flex: 1, position: 'relative' }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
};

export default TrafficTimeline;
