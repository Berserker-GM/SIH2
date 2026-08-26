import { useData } from '../context/DataProvider';
import { ReactFlow, Background, Controls } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

export default function AttackGraph() {
  const { incidentGraph } = useData();

  if (!incidentGraph) {
    return (
      <div className="panel h-full flex flex-col items-center justify-center relative overflow-hidden">
        {/* Animated Sweeping Radar Effect */}
        <div className="absolute inset-0 flex items-center justify-center opacity-30">
          <div className="w-[300px] h-[300px] rounded-full border border-[rgba(0,240,255,0.2)] relative flex items-center justify-center">
            <div className="w-[200px] h-[200px] rounded-full border border-[rgba(0,240,255,0.3)]"></div>
            <div className="w-[100px] h-[100px] rounded-full border border-[rgba(0,240,255,0.4)] absolute"></div>
            {/* Sweeper */}
            <div 
              className="absolute w-1/2 h-[2px] bg-gradient-to-r from-transparent to-[#00F0FF] origin-left animate-spin" 
              style={{ top: '50%', left: '50%', animationDuration: '3s' }}
            ></div>
          </div>
        </div>
        
        <div className="z-10 flex flex-col items-center gap-3">
          <div className="flex gap-1 mb-2">
            <div className="w-2 h-2 bg-[#00F0FF] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 bg-[#00F0FF] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 bg-[#00F0FF] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
          <p className="text-[#00F0FF] font-mono text-sm tracking-[0.3em] font-bold">
            AWAITING SIGNAL
          </p>
          <p className="text-[#4A5568] font-mono text-[10px] tracking-widest uppercase">
            Select incident from feed to map attack vector
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel h-full relative overflow-hidden flex flex-col">
      <div className="panel-header absolute top-0 left-0 right-0 z-10 bg-[rgba(0,0,0,0.8)] backdrop-blur-md">
        <span className="text-[#00F0FF] mr-2 animate-pulse">⎈</span> ATTACK PROGRESSION GRAPH
      </div>
      <div className="flex-1 w-full h-full pt-10">
        <ReactFlow 
          nodes={incidentGraph.nodes} 
          edges={incidentGraph.edges}
          fitView
          className="dark-theme"
        >
          <Background color="#00F0FF" gap={20} opacity={0.05} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
}
