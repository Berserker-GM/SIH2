import { useState, useMemo } from 'react';
import { useData } from '../context/DataProvider';
import Header from '../components/Header';
import SummaryStats from '../components/SummaryStats';
import SeverityBreakdown from '../components/SeverityBreakdown';
import ModelMetrics from '../components/ModelMetrics';
import AttackerLeaderboard from '../components/AttackerLeaderboard';
import EventTicker from '../components/EventTicker';
import ThreatMap from '../components/ThreatMap';
import AttackTimeline from '../components/AttackTimeline';
import IncidentFeed from '../components/IncidentFeed';
import IncidentDetail from '../components/IncidentDetail';
import RiskChart from '../components/RiskChart';
import TechniqueChart from '../components/TechniqueChart';
import EventLogTable from '../components/EventLogTable';
import WorldModelPanel from '../components/WorldModelPanel';

export default function UnifiedDashboard() {
  const { 
    connected, incidents, metrics, tickerEvents, events,
    selectedIncidentId, setSelectedIncidentId, selectedIncidentDetail, incidentGraph
  } = useData();
  
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const selectedIncident = useMemo(() => {
    return selectedIncidentDetail || incidents.find((inc) => inc.incident_id === selectedIncidentId) || null;
  }, [incidents, selectedIncidentId, selectedIncidentDetail]);

  return (
    <div className="flex-1 min-w-0 w-full flex flex-col font-sans text-white bg-transparent h-screen overflow-hidden">
      {/* Top Header */}
      <Header connected={connected} />

      {!connected && (
        <div className="bg-[rgba(255,68,68,0.1)] border-b border-[var(--color-critical)] text-[var(--color-critical)] text-[10px] font-bold px-4 py-1.5 text-center tracking-[0.2em] uppercase flex items-center justify-center gap-2 flex-none z-50">
          <span className="w-2 h-2 rounded-full bg-[var(--color-critical)] animate-ping"></span>
          CONNECTION OFFLINE — RECONNECTING IN 3S
        </div>
      )}

      {/* Main Grid Workspace */}
      <main className="flex-1 min-w-0 overflow-hidden p-4 gap-4 flex flex-col min-h-0">
        
        {/* Mobile Hamburger Header (Visible only on mobile/tablet) */}
        <div className="xl:hidden flex items-center justify-between bg-[var(--color-bg-secondary)] border border-[var(--color-border)] p-3 rounded-lg flex-none">
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-black tracking-widest text-[var(--color-accent)]">NORMATIVE</span>
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--color-critical)] animate-pulse"></span>
          </div>
          <button 
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="text-white bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] p-2 rounded border border-[var(--color-border)] text-xs"
          >
            {mobileMenuOpen ? 'CLOSE CONTROLS' : 'SIDEBARS'}
          </button>
        </div>

        {/* 3-Column Layout Container */}
        <div className="flex-1 min-w-0 min-h-0 grid grid-cols-1 xl:grid-cols-10 gap-4 relative">
          
          {/* COLUMN 1: LEFT SIDEBAR (20% - xl:col-span-2) */}
          <div className={`
            ${mobileMenuOpen ? 'absolute inset-0 z-40 bg-[#0D1117] flex' : 'hidden'} 
            xl:flex xl:relative xl:col-span-2 min-w-0 flex-col gap-4 overflow-y-auto pr-1 custom-scrollbar min-h-0 h-full
          `}>
            {/* Logo block inside Left Sidebar */}
            <div className="flex items-center justify-between bg-[rgba(0,212,255,0.02)] border border-[var(--color-border)] p-3 rounded-lg flex-none">
              <span className="font-mono text-sm font-black text-white tracking-widest">
                NORMATIVE LOGO
              </span>
              <span className="flex h-2.5 w-2.5 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--color-critical)] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[var(--color-critical)]"></span>
              </span>
            </div>

            {/* Widgets */}
            <div className="flex-none">
              <SummaryStats incidents={incidents} metrics={metrics} />
            </div>
            
            <div className="flex-none">
              <SeverityBreakdown incidents={incidents} />
            </div>

            <div className="flex-none">
              <TechniqueChart incidents={incidents} />
            </div>

            <div className="flex-none">
              <ModelMetrics metrics={metrics} />
            </div>

            <div className="flex-none">
              <AttackerLeaderboard incidents={incidents} />
            </div>
          </div>

          {/* COLUMN 2: CENTER COLUMN (50% - xl:col-span-5) */}
          <div className="xl:col-span-5 min-w-0 flex flex-col gap-4 overflow-y-auto min-h-0 h-full custom-scrollbar">
            {/* Live Ticker */}
            <div className="flex-none">
              <EventTicker events={tickerEvents} />
            </div>

            <div className="flex-none">
              <WorldModelPanel />
            </div>

            {/* Threat Map */}
            <div className="flex-none h-[350px]">
              <div className="panel h-full overflow-hidden relative border-[var(--color-accent)] shadow-[0_0_15px_rgba(0,212,255,0.1)]">
                <ThreatMap 
                  incidents={incidents} 
                  selectedIncidentId={selectedIncidentId} 
                  onSelectIncident={setSelectedIncidentId} 
                />
              </div>
            </div>

            {/* Attack Timeline */}
            <div className="flex-none">
              <AttackTimeline incident={selectedIncident} />
            </div>

            {/* Risk score over time chart */}
            <div className="flex-none">
              <RiskChart incidents={incidents} onSelectIncident={setSelectedIncidentId} />
            </div>
          </div>

          {/* COLUMN 3: RIGHT SIDEBAR (30% - xl:col-span-3) */}
          <div className="xl:col-span-3 min-w-0 flex flex-col gap-4 overflow-y-auto min-h-0 h-full custom-scrollbar pr-1">
            {/* Incident Feed */}
            <div className="flex-none h-[250px]">
              <IncidentFeed 
                incidents={incidents} 
                selectedIncidentId={selectedIncidentId}
                onSelectIncident={setSelectedIncidentId} 
              />
            </div>

            {/* Selected Incident Detail */}
            <div className="flex-1 min-h-[300px]">
              <IncidentDetail incident={selectedIncident} />
            </div>
          </div>

        </div>

        {/* BOTTOM: Full Width Live Event Log Table */}
        <div className="flex-none h-[200px] min-h-[180px] z-10">
          <EventLogTable events={events} />
        </div>

      </main>
    </div>
  );
}
