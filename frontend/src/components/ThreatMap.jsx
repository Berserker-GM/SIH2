import { useMemo } from 'react';
import { MapContainer, GeoJSON, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import geoData from './countries.geo.json';

const SEVERITY_COLORS = {
  critical: '#FF4444',
  high: '#FF8C00',
  medium: '#FFD700',
  low: '#00FF88',
};

const IP_LOCATIONS = {
  '203.0.113.99': { lat: 39.9042, lng: 116.4074, city: 'Beijing' },
  '45.33.32.156': { lat: 37.5485, lng: -121.9886, city: 'Fremont, CA' },
  '198.51.100.23': { lat: 51.5074, lng: -0.1278, city: 'London' },
  '185.220.101.1': { lat: 52.52, lng: 13.405, city: 'Berlin' },
  '91.219.237.22': { lat: 59.3293, lng: 18.0686, city: 'Stockholm' },
};

function isInternalIP(ip) {
  return ip.startsWith('192.168.') || ip.startsWith('10.');
}

export default function ThreatMap({
  incidents = [],
  selectedIncidentId,
  onSelectIncident,
}) {
  const { plotted, internalCount } = useMemo(() => {
    const plotted = [];
    let internalCount = 0;

    incidents.forEach((incident) => {
      const assets = incident.affected_assets || [];
      let hasExternal = false;

      assets.forEach((ip) => {
        if (isInternalIP(ip)) {
          internalCount++;
          return;
        }
        const loc = IP_LOCATIONS[ip];
        if (loc) {
          hasExternal = true;
          // Deduplicate mapped markers for clean rendering
          if (!plotted.some(p => p.ip === ip && p.incident.incident_id === incident.incident_id)) {
            plotted.push({
              ...loc,
              incident,
              ip,
            });
          }
        }
      });
      if (!hasExternal) return;
    });

    return { plotted, internalCount };
  }, [incidents]);

  const severityColor = (severity) => SEVERITY_COLORS[severity] || SEVERITY_COLORS.low;

  return (
    <div className="flex flex-col h-full w-full relative p-1">
      <div className="flex-1 w-full rounded-lg overflow-hidden border border-[#00D4FF] cyber-grid-bg relative z-0 min-h-[300px] shadow-[0_0_15px_rgba(0,212,255,0.15)]">
        <MapContainer
          center={[20, 0]}
          zoom={2}
          minZoom={1.5}
          maxZoom={10}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
          attributionControl={false}
        >
          <GeoJSON 
            data={geoData} 
            style={{
              fillColor: '#050F1D',
              weight: 1.5,
              color: 'rgba(0, 212, 255, 0.6)',
              fillOpacity: 1,
            }} 
          />

          {plotted.map((p, idx) => {
            const color = severityColor(p.incident.severity);
            const isSelected = p.incident.incident_id === selectedIncidentId;
            // Radius = risk_score / 5
            const radius = Math.max(p.incident.risk_score / 5, 4);

            return (
              <CircleMarker
                key={`${p.incident.incident_id}-${p.ip}-${idx}`}
                center={[p.lat, p.lng]}
                radius={isSelected ? radius * 1.5 : radius}
                fillColor={color}
                color={color}
                weight={isSelected ? 4 : 0}
                fillOpacity={1}
                pathOptions={{ className: 'glowing-marker' }}
                eventHandlers={{
                  click: () => {
                    onSelectIncident?.(p.incident.incident_id);
                  },
                }}
              >
                <Popup className="custom-map-popup">
                  <div className="bg-[#161B22] text-[#F0F6FC] border border-[var(--color-border)] p-2.5 rounded shadow-lg text-[11px] font-mono leading-relaxed">
                    <div className="font-bold text-[var(--color-accent)] mb-1">
                      Incident: {p.incident.incident_id}
                    </div>
                    <div>IP: {p.ip} ({p.city})</div>
                    <div>Risk Score: <span className="font-bold text-white">{p.incident.risk_score}</span></div>
                    {p.incident.attack_chain && p.incident.attack_chain.length > 0 && (
                      <div className="mt-1 border-t border-[rgba(255,255,255,0.1)] pt-1 text-gray-400">
                        Technique: {p.incident.attack_chain[0].technique_name}
                      </div>
                    )}
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {internalCount > 0 && (
          <div className="absolute bottom-4 left-4 bg-[rgba(3,8,17,0.85)] border border-[#00D4FF] rounded px-3 py-1.5 text-center animate-fade-in shadow-[0_0_15px_rgba(0,212,255,0.3)] z-[1000] pointer-events-auto">
            <span className="text-[#00D4FF] text-[10px] font-mono tracking-widest uppercase">
              Internal Threats:
            </span>
            <span className="text-[#00FF88] text-sm font-bold font-mono ml-2">
              {internalCount}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
