import React, { useState, useRef, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';

const CustomNode = ({ data, id }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 200, height: 80 });
  const nodeRef = useRef(null);
  const resizeRef = useRef(null);

  // Toggle collapse state
  const toggleCollapse = useCallback(() => {
    setIsCollapsed(!isCollapsed);
  }, [isCollapsed]);

  // Handle mouse down for resizing
  const handleResizeMouseDown = useCallback((e) => {
    e.stopPropagation();
    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = dimensions.width;
    const startHeight = dimensions.height;

    const handleMouseMove = (e) => {
      const newWidth = Math.max(100, startWidth + (e.clientX - startX));
      const newHeight = Math.max(50, startHeight + (e.clientY - startY));
      setDimensions({ width: newWidth, height: newHeight });
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [dimensions]);

  return (
    <div 
      ref={nodeRef}
      className={`custom-node ${isCollapsed ? 'collapsed' : ''}`}
      style={{
        width: isCollapsed ? 120 : dimensions.width,
        height: isCollapsed ? 40 : dimensions.height,
        backgroundColor: '#fff',
        border: '1px solid #777',
        borderRadius: 5,
        padding: 10,
        position: 'relative',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        transition: 'all 0.2s ease'
      }}
      title={data.label} // Tooltip using title attribute
    >
      {/* Collapse/Expand button */}
      <button 
        onClick={toggleCollapse}
        style={{
          position: 'absolute',
          top: 2,
          right: 2,
          background: 'none',
          border: 'none',
          fontSize: 12,
          cursor: 'pointer',
          padding: 2,
          borderRadius: 3
        }}
      >
        {isCollapsed ? '+' : '-'}
      </button>

      {/* Node content */}
      <div style={{ 
        overflow: 'hidden', 
        textOverflow: 'ellipsis',
        whiteSpace: isCollapsed ? 'nowrap' : 'normal'
      }}>
        {isCollapsed ? (
          <strong>{data.label}</strong>
        ) : (
          <>
            <div style={{ fontWeight: 'bold', marginBottom: 5 }}>{data.label}</div>
            {data.description && (
              <div style={{ fontSize: 12, color: '#555' }}>{data.description}</div>
            )}
          </>
        )}
      </div>

      {/* Resize handle */}
      {!isCollapsed && (
        <div
          ref={resizeRef}
          onMouseDown={handleResizeMouseDown}
          style={{
            position: 'absolute',
            bottom: 2,
            right: 2,
            width: 10,
            height: 10,
            backgroundColor: '#777',
            cursor: 'se-resize',
            transform: 'rotate(45deg)'
          }}
        />
      )}

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#555' }}
        onConnect={(params) => console.log('handle onConnect', params)}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#555' }}
      />
    </div>
  );
};

export default CustomNode;
