import { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { generateMockData, fetchMockIncidentDetail, fetchMockIncidentGraph } from './mockData';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/live';

const DataContext = createContext(null);

export function useData() {
  return useContext(DataContext);
}

export function DataProvider({ children }) {
  const [incidents, setIncidents] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [events, setEvents] = useState([]);
  const [tickerEvents, setTickerEvents] = useState([]);
  const [selectedIncidentId, setSelectedIncidentId] = useState(null);
  const [selectedIncidentDetail, setSelectedIncidentDetail] = useState(null);
  const [incidentGraph, setIncidentGraph] = useState(null);
  const [connected, setConnected] = useState(false);
  const [useMock, setUseMock] = useState(false);

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const metricsIntervalRef = useRef(null);

  const fetchInitialData = useCallback(async () => {
    try {
      const [incRes, metRes] = await Promise.all([
        axios.get(`${API_BASE}/api/incidents`),
        axios.get(`${API_BASE}/api/metrics`),
      ]);
      setIncidents(incRes.data.slice(0, 50));
      setMetrics(metRes.data);
      setConnected(true);

      const ticker = [];
      incRes.data.forEach(inc => {
        (inc.attack_chain || []).forEach(step => {
          ticker.push({
            timestamp: step.timestamp,
            technique_id: step.technique_id,
            technique_name: step.technique_name,
          });
        });
      });
      ticker.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      setTickerEvents(ticker.slice(0, 20));

      try {
        const evtRes = await axios.get(`${API_BASE}/api/events`);
        setEvents(evtRes.data.slice(-100));
      } catch {
        // silently fail
      }
    } catch {
      console.log('Backend not available, using mock data');
      const mock = generateMockData();
      setIncidents(mock.incidents);
      setMetrics(mock.metrics);
      setEvents(mock.events);
      setTickerEvents(mock.tickerEvents);
      setUseMock(true);
      setConnected(true);
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    if (useMock) return;
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // The backend sends the raw incident payload directly
          if (data.type === 'incident' || data.incident_id) {
            const payload = data.type === 'incident' ? data.payload : data;
            
            setIncidents(prev => {
              // Deduplicate if incident is already in feed
              if (prev.some(inc => inc.incident_id === payload.incident_id)) {
                return prev;
              }
              return [payload, ...prev].slice(0, 50);
            });
            
            if (payload.attack_chain) {
              const newTicker = payload.attack_chain.map(step => ({
                timestamp: step.timestamp,
                technique_id: step.technique_id,
                technique_name: step.technique_name,
              }));
              setTickerEvents(prev => [...newTicker, ...prev].slice(0, 20));
            }
          } else if (data.type === 'event' || data.event_id) {
            const payload = data.type === 'event' ? data.payload : data;
            setEvents(prev => [...prev, payload].slice(-100));
          }
        } catch (e) {
          console.error('WebSocket parse error', e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimerRef.current = setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = () => ws.close();
    } catch {
      setConnected(false);
    }
  }, [useMock]);

  const pollMetrics = useCallback(async () => {
    if (useMock) {
      setMetrics(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          total_events_processed: (prev.total_events_processed || 0) + Math.floor(Math.random() * 500),
          total_anomalies_detected: (prev.total_anomalies_detected || 0) + (Math.random() > 0.7 ? 1 : 0),
          fpr: Math.max(0, (prev.fpr || 0) + (Math.random() - 0.5) * 0.3),
        };
      });
      return;
    }
    try {
      const res = await axios.get(`${API_BASE}/api/metrics`);
      setMetrics(res.data);
    } catch {}
  }, [useMock]);

  useEffect(() => {
    if (!useMock) return;
    const TECHNIQUES = [
      { id: 'T1110', name: 'Brute Force' }, { id: 'T1059', name: 'Command Execution' },
      { id: 'T1053', name: 'Scheduled Task' }, { id: 'T1021', name: 'Remote Services' },
      { id: 'T1486', name: 'Data Encrypted' }, { id: 'T1071', name: 'App Layer Protocol' },
    ];
    const SOURCE_TYPES = ['syslog', 'netflow', 'sysmon', 'auth'];
    const IPS = ['203.0.113.99', '45.33.32.156', '198.51.100.23', '192.168.1.10', '10.0.0.50'];
    const ACTIONS = ['SSH login attempt', 'HTTP GET /admin', 'Process spawned', 'DNS query', 'Port scan'];

    const interval = setInterval(() => {
      const hasMitre = Math.random() > 0.5;
      const tech = TECHNIQUES[Math.floor(Math.random() * TECHNIQUES.length)];
      const newEvent = {
        timestamp: new Date().toISOString(),
        source_type: SOURCE_TYPES[Math.floor(Math.random() * SOURCE_TYPES.length)],
        src_ip: IPS[Math.floor(Math.random() * IPS.length)],
        action: ACTIONS[Math.floor(Math.random() * ACTIONS.length)],
        status: Math.random() > 0.3 ? 'success' : 'blocked',
        mitre_tag: hasMitre ? tech.id : null,
      };
      setEvents(prev => [...prev, newEvent].slice(-100));

      if (hasMitre) {
        setTickerEvents(prev => [{
          timestamp: newEvent.timestamp,
          technique_id: tech.id,
          technique_name: tech.name,
        }, ...prev].slice(0, 20));
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [useMock]);

  useEffect(() => {
    fetchInitialData();
  }, [fetchInitialData]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    };
  }, [connectWebSocket]);

  const pollEvents = useCallback(async () => {
    if (useMock) return;
    try {
      const res = await axios.get(`${API_BASE}/api/events`);
      setEvents(res.data.slice(-100));
    } catch (e) {
      console.error("Failed to poll events", e);
    }
  }, [useMock]);

  useEffect(() => {
    const eventInterval = setInterval(pollEvents, 3000);
    return () => clearInterval(eventInterval);
  }, [pollEvents]);

  useEffect(() => {
    metricsIntervalRef.current = setInterval(pollMetrics, 10000);
    return () => {
      if (metricsIntervalRef.current) clearInterval(metricsIntervalRef.current);
    };
  }, [pollMetrics]);

  // Fetch detail and graph when selected incident changes
  useEffect(() => {
    if (!selectedIncidentId) {
      setSelectedIncidentDetail(null);
      setIncidentGraph(null);
      return;
    }

    const fetchDetail = async () => {
      if (useMock) {
        const detail = await fetchMockIncidentDetail(selectedIncidentId, incidents);
        const graph = await fetchMockIncidentGraph(selectedIncidentId);
        setSelectedIncidentDetail(detail);
        setIncidentGraph(graph);
      } else {
        try {
          const [detailRes, graphRes] = await Promise.all([
            axios.get(`${API_BASE}/api/incidents/${selectedIncidentId}`),
            axios.get(`${API_BASE}/api/graph/${selectedIncidentId}`),
          ]);
          setSelectedIncidentDetail(detailRes.data);
          setIncidentGraph(graphRes.data);
        } catch (e) {
          console.error("Failed to fetch detailed incident data", e);
        }
      }
    };

    fetchDetail();
  }, [selectedIncidentId, useMock, incidents]);

  const value = {
    incidents,
    metrics,
    events,
    tickerEvents,
    selectedIncidentId,
    setSelectedIncidentId,
    selectedIncidentDetail,
    incidentGraph,
    connected
  };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}
