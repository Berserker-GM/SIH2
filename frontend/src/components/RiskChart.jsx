import { useMemo, useCallback } from 'react';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';

const SEVERITY_COLORS = {
  critical: '#FF4444',
  high: '#FF8C00',
  medium: '#FFD700',
  low: '#00FF88',
};

function CustomDot({ cx, cy, payload, onSelectIncident }) {
  const color = SEVERITY_COLORS[payload.severity] || SEVERITY_COLORS.low;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={4}
      fill={color}
      stroke={color}
      strokeWidth={2}
      style={{ cursor: 'pointer' }}
      onClick={() => onSelectIncident?.(payload.incident_id)}
    />
  );
}

function ActiveDot({ cx, cy, payload }) {
  const color = SEVERITY_COLORS[payload.severity] || SEVERITY_COLORS.low;
  return (
    <g>
      <circle
        cx={cx}
        cy={cy}
        r={10}
        fill={color}
        fillOpacity={0.2}
        stroke="none"
      />
      <circle
        cx={cx}
        cy={cy}
        r={6}
        fill={color}
        stroke={color}
        strokeWidth={2}
      />
    </g>
  );
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.[0]) return null;
  const data = payload[0].payload;
  const color = SEVERITY_COLORS[data.severity] || SEVERITY_COLORS.low;

  return (
    <div className="bg-[#21262D] border border-[#30363D] rounded-lg shadow-lg p-3 min-w-[160px]">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-[family-name:var(--font-mono)] text-[#00D4FF]">
          {data.incident_id}
        </span>
      </div>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg font-bold font-[family-name:var(--font-mono)]" style={{ color }}>
          {data.risk_score}
        </span>
        <span
          className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded"
          style={{
            color,
            backgroundColor: `${color}15`,
          }}
        >
          {data.severity}
        </span>
      </div>
      <div className="text-[10px] text-[#6E7681] font-[family-name:var(--font-mono)]">
        {data.time}
      </div>
    </div>
  );
}

export default function RiskChart({ incidents = [], onSelectIncident }) {
  const chartData = useMemo(() => {
    return incidents.slice(-20).map((inc) => ({
      time: format(new Date(inc.detected_at), 'HH:mm'),
      risk_score: inc.risk_score,
      severity: inc.severity,
      incident_id: inc.incident_id,
    }));
  }, [incidents]);

  const renderDot = useCallback(
    (props) => (
      <CustomDot
        key={`dot-${props.index}`}
        {...props}
        onSelectIncident={onSelectIncident}
      />
    ),
    [onSelectIncident]
  );

  const renderActiveDot = useCallback(
    (props) => <ActiveDot key={`active-${props.index}`} {...props} />,
    []
  );

  return (
    <div className="panel">
      <div className="panel-header">RISK SCORE TIMELINE</div>
      <div className="pt-2">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart
            data={chartData}
            margin={{ top: 8, right: 16, left: -8, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#30363D" />
            <XAxis
              dataKey="time"
              tick={{ fill: '#6E7681', fontSize: 10 }}
              tickLine={{ stroke: '#30363D' }}
              axisLine={{ stroke: '#30363D' }}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: '#6E7681', fontSize: 10 }}
              tickLine={{ stroke: '#30363D' }}
              axisLine={{ stroke: '#30363D' }}
            />
            <ReferenceLine
              y={60}
              stroke="#FF4444"
              strokeDasharray="8 4"
              label={{
                value: 'THRESHOLD',
                position: 'insideTopRight',
                fill: '#FF4444',
                fontSize: 10,
                fontWeight: 600,
              }}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ stroke: '#30363D', strokeDasharray: '4 2' }}
            />
            <Line
              type="monotone"
              dataKey="risk_score"
              stroke="#00D4FF"
              strokeWidth={2}
              dot={renderDot}
              activeDot={renderActiveDot}
              animationDuration={800}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
