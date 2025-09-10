'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Menu, Button, Text, rem, Badge, Group } from '@mantine/core';
import { IconX } from '@tabler/icons-react';

interface Selection {
  type: 'key' | 'value';
  number: number;
  text: string;
  range: Range;
}

interface InteractiveRecordModalProps {
  content: any; // Change to any to handle any type of content
  onClose: () => void;
}

// Helper function to safely convert any value to a string
const safeString = (value: any): string => {
  if (value === null || value === undefined) {
    return '';
  }
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2);
    } catch (e) {
      console.error('Error stringifying object value:', e);
      return '[Complex Object]';
    }
  }
  return String(value);
};

const InteractiveRecordModal: React.FC<InteractiveRecordModalProps> = ({ content, onClose }) => {
  // Convert content to a safe string at the component level
  const safeContent = safeString(content);

  const [selections, setSelections] = useState<Selection[]>([]);
  const [currentSelection, setCurrentSelection] = useState<{
    text: string;
    range: Range;
  } | null>(null);

  const contentRef = useRef<HTMLDivElement>(null);

  const handleTextHighlight = useCallback(() => {
    const selection = window.getSelection();
    if (selection && !selection.isCollapsed && contentRef.current?.contains(selection.anchorNode)) {
      const range = selection.getRangeAt(0);
      setCurrentSelection({ text: selection.toString(), range });
    } else {
      setCurrentSelection(null);
    }
  }, []);

  // Simplified to only handle keys - values are implicit
  const handleSelection = () => {
    if (!currentSelection) return;
    const number = selections.length + 1;
    setSelections((prev) => [
      ...prev,
      {
        type: 'key',
        number,
        text: currentSelection.text,
        range: currentSelection.range,
      },
    ]);
    setCurrentSelection(null);
    window.getSelection()?.removeAllRanges();
  };

  const removeSelection = (index: number) => {
    setSelections((prev) => {
      const newSelections = prev.filter((_, i) => i !== index);
      return newSelections.map((selection, i) => ({
        ...selection,
        number: newSelections.filter((s, j) => s.type === selection.type && j <= i).length,
      }));
    });
  };

  const highlightSelections = useCallback(() => {
    if (!contentRef.current) return;

    // Remove existing highlights
    contentRef.current.querySelectorAll('mark').forEach((mark) => {
      const parent = mark.parentNode;
      if (parent) {
        parent.insertBefore(document.createTextNode(mark.textContent || ''), mark);
        parent.removeChild(mark);
      }
    });

    // Only highlight keys
    selections.forEach((selection) => {
      const mark = document.createElement('mark');
      mark.style.backgroundColor = 'rgba(255,0,0,0.2)';
      selection.range.surroundContents(mark);
    });
  }, [selections]);

  useEffect(() => {
    highlightSelections();
  }, [highlightSelections]);

  // We can also style our <pre> area
  const preStyles: React.CSSProperties = {
    fontFamily: 'inherit',
    fontWeight: 300,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    width: '100%',
    margin: 0,
    padding: rem(8),
    backgroundColor: 'lavender',
    borderRadius: rem(4),
    border: '1px solid black',
    overflowX: 'hidden',
  };

  return (
    <div style={{ position: 'relative', overflow: 'hidden' }}>
      <Group mb="md" p="xs">
        {selections.map((selection, index) => (
          <Badge
            key={index}
            color="red"
            rightSection={
              <IconX
                size="0.8rem"
                style={{ cursor: 'pointer' }}
                onClick={() => removeSelection(index)}
              />
            }
          >
            {`Key ${selection.number}: ${selection.text}`}
          </Badge>
        ))}
      </Group>

      <div
        ref={contentRef}
        style={{ marginTop: '20px', ...preStyles }}
        onMouseUp={handleTextHighlight}
        onTouchEnd={handleTextHighlight}
      >
        {safeContent}
      </div>

      {/* Simplified menu - only "Select as Key" option */}
      {currentSelection && (
        <Menu
          opened={!!currentSelection}
          onClose={() => setCurrentSelection(null)}
          withArrow
          position="bottom-start"
          styles={(theme) => ({
            dropdown: {
              left: `${currentSelection.range.getBoundingClientRect().left + window.scrollX}px`,
              top: `${currentSelection.range.getBoundingClientRect().bottom + window.scrollY}px`,
            },
          })}
        >
          <Menu.Target>
            <Button size="xs" style={{ visibility: 'hidden' }}>
              Select as Key
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Item onClick={() => handleSelection()}>
              Select as Key {selections.length + 1}
            </Menu.Item>
          </Menu.Dropdown>
        </Menu>
      )}

      <Group justify="flex-end" mt="md">
        <Button onClick={onClose}>Close</Button>
      </Group>
    </div>
  );
};

export default InteractiveRecordModal;
