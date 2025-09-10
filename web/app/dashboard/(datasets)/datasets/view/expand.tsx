'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Menu, Button, Text, rem, Badge, Group, MultiSelect, ActionIcon, Stack, Divider, ScrollArea, Tooltip, TextInput, Switch, Card } from '@mantine/core';
import { IconX, IconChevronRight, IconChevronDown, IconCopy } from '@tabler/icons-react';

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
  // Try to parse JSON; otherwise fall back to text mode
  const [jsonRoot, setJsonRoot] = useState<any | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  useEffect(() => {
    try {
      if (typeof content === 'string') {
        setJsonRoot(JSON.parse(content));
      } else if (typeof content === 'object' && content !== null) {
        setJsonRoot(content);
      } else {
        setJsonRoot(null);
      }
      setParseError(null);
    } catch (e: any) {
      setJsonRoot(null);
      setParseError(e?.message || 'Invalid JSON');
    }
  }, [content]);

  // -------- JSON Tree UI --------
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState('');
  const [selectedTopKeys, setSelectedTopKeys] = useState<string[]>([]);
  const [showOnlySelected, setShowOnlySelected] = useState(false);

  const toggle = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const expandAll = () => {
    const all = new Set<string>();
    const walk = (node: any, path: string) => {
      all.add(path);
      if (node && typeof node === 'object') {
        if (Array.isArray(node)) {
          node.forEach((v, i) => walk(v, `${path}[${i}]`));
        } else {
          Object.keys(node).forEach((k) => walk(node[k], path ? `${path}.${k}` : k));
        }
      }
    };
    walk(jsonRoot, 'root');
    setExpanded(all);
  };

  const collapseAll = () => setExpanded(new Set());

  const matchesFilter = (keyPath: string, value: any) => {
    const f = filter.trim().toLowerCase();
    if (!f) return true;
    if (keyPath.toLowerCase().includes(f)) return true;
    try {
      const s = typeof value === 'string' ? value : JSON.stringify(value);
      return s?.toLowerCase().includes(f);
    } catch {
      return false;
    }
  };

  const copyToClipboard = (text: string) => {
    try {
      navigator.clipboard?.writeText(text);
    } catch {}
  };

  const NodeRow: React.FC<{ k?: string; value: any; path: string; depth: number }> = ({ k, value, path, depth }) => {
    const isObj = value && typeof value === 'object';
    const isArr = Array.isArray(value);
    const label = k ?? (isArr ? '[]' : '{}');
    const currentPath = path;
    const open = expanded.has(currentPath);
    const show = matchesFilter(currentPath, value);
    if (!show) return null;

    const leftPad = { paddingLeft: rem(8 + depth * 12) } as React.CSSProperties;

    return (
      <div>
        <Group wrap="nowrap" gap="xs" style={leftPad}>
          {isObj ? (
            <ActionIcon size="sm" variant="subtle" onClick={() => toggle(currentPath)}>
              {open ? <IconChevronDown size="1rem" /> : <IconChevronRight size="1rem" />}
            </ActionIcon>
          ) : (
            <div style={{ width: rem(28) }} />
          )}
          {k && (
            <Group gap={4} align="center" style={{ minWidth: rem(120) }}>
              <Text fw={500} size="sm">
                {k}
              </Text>
              <Tooltip label="Copy path">
                <ActionIcon size="xs" variant="subtle" onClick={() => copyToClipboard(currentPath)}>
                  <IconCopy size="0.8rem" />
                </ActionIcon>
              </Tooltip>
            </Group>
          )}
          {!isObj && (
            <>
              <Text size="sm" c="dimmed" style={{ flex: 1 }}>
                {safeString(value)}
              </Text>
              <Tooltip label="Copy value">
                <ActionIcon size="sm" variant="subtle" onClick={() => copyToClipboard(safeString(value))}>
                  <IconCopy size="0.9rem" />
                </ActionIcon>
              </Tooltip>
            </>
          )}
        </Group>
        {isObj && open && (
          <div>
            {isArr
              ? (value as any[]).map((v, i) => (
                  <NodeRow key={i} k={`[${i}]`} value={v} path={`${currentPath}[${i}]`} depth={depth + 1} />
                ))
              : Object.entries(value as Record<string, any>).map(([ck, cv]) => (
                  <NodeRow key={ck} k={ck} value={cv} path={`${currentPath}.${ck}`} depth={depth + 1} />
                ))}
          </div>
        )}
      </div>
    );
  };

  // -------- Text fallback (original selection highlighter) --------
  const safeContent = safeString(content);
  const [selections, setSelections] = useState<Selection[]>([]);
  const [currentSelection, setCurrentSelection] = useState<{ text: string; range: Range } | null>(null);
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

  const handleSelection = () => {
    if (!currentSelection) return;
    const number = selections.length + 1;
    setSelections((prev) => [
      ...prev,
      { type: 'key', number, text: currentSelection.text, range: currentSelection.range },
    ]);
    setCurrentSelection(null);
    window.getSelection()?.removeAllRanges();
  };

  const removeSelection = (index: number) => {
    setSelections((prev) => {
      const next = prev.filter((_, i) => i !== index);
      return next.map((sel, i) => ({
        ...sel,
        number: next.filter((s, j) => s.type === sel.type && j <= i).length,
      }));
    });
  };

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

  const isJson = jsonRoot !== null;

  // Unified return: keep hooks order stable across renders
  return (
    <div style={{ position: 'relative', overflow: 'hidden' }}>
      {isJson ? (
        <>
          {(() => {
            const topKeys: string[] = Array.isArray(jsonRoot)
              ? (jsonRoot as any[]).map((_, i) => `[${i}]`)
              : Object.keys(jsonRoot || {});
            const visibleTopKeys = showOnlySelected && selectedTopKeys.length > 0 ? selectedTopKeys : topKeys;
            return (
              <>
                <Group justify="space-between" mb="sm">
                  <Group gap="xs">
                    <Button variant="light" onClick={expandAll}>Expand all</Button>
                    <Button variant="light" onClick={collapseAll}>Collapse all</Button>
                  </Group>
                  <Group gap="xs">
                    <TextInput placeholder="Filter by key or value" value={filter} onChange={(e) => setFilter(e.currentTarget.value)} w={240} />
                    <Switch checked={showOnlySelected} onChange={(e) => setShowOnlySelected(e.currentTarget.checked)} label="Only selected keys" />
                  </Group>
                </Group>

                <Group align="flex-start" gap="md" wrap="nowrap">
                  <div style={{ minWidth: 260 }}>
                    <Text fw={500} size="sm" mb={4}>Top-level keys</Text>
                    <MultiSelect
                      data={topKeys.map((k) => ({ value: k, label: k }))}
                      value={selectedTopKeys}
                      onChange={setSelectedTopKeys}
                      searchable
                      placeholder="Select keys to focus"
                      clearable
                      maxDropdownHeight={240}
                    />
                  </div>

                  <ScrollArea style={{ flex: 1, maxHeight: '60vh' }}>
                    <Stack gap="xs">
                      {visibleTopKeys.length === 0 && (
                        <Text size="sm" c="dimmed">No keys selected</Text>
                      )}
                      {visibleTopKeys.map((k) => {
                        const value = Array.isArray(jsonRoot)
                          ? (jsonRoot as any[])[parseInt(k.replace(/\[(\d+)\]/, '$1'))]
                          : (jsonRoot as any)[k];
                        const path = `root.${k}`;
                        const isObj = value && typeof value === 'object';
                        return (
                          <Card key={k} withBorder radius="sm" p="sm">
                            <Group gap="xs">
                              <Text fw={600}>{k}</Text>
                              {isObj && (
                                <Button size="xs" variant="light" onClick={() => toggle(path)}>
                                  {expanded.has(path) ? 'Collapse' : 'Expand'}
                                </Button>
                              )}
                            </Group>
                            <Divider my={6} />
                            <NodeRow k={k} value={value} path={path} depth={0} />
                          </Card>
                        );
                      })}
                    </Stack>
                  </ScrollArea>
                </Group>
              </>
            );
          })()}
        </>
      ) : (
        <>
          <Group mb="md" p="xs">
            {selections.map((selection, index) => (
              <Badge
                key={index}
                color="red"
                rightSection={<IconX size="0.8rem" style={{ cursor: 'pointer' }} onClick={() => removeSelection(index)} />}
              >
                {`Key ${selection.number}: ${selection.text}`}
              </Badge>
            ))}
          </Group>

          <div ref={contentRef} style={{ marginTop: '20px', ...preStyles }} onMouseUp={handleTextHighlight} onTouchEnd={handleTextHighlight}>
            {safeContent}
          </div>

          {currentSelection && (
            <Menu
              opened={!!currentSelection}
              onClose={() => setCurrentSelection(null)}
              withArrow
              position="bottom-start"
              styles={{
                dropdown: {
                  left: `${currentSelection.range.getBoundingClientRect().left + window.scrollX}px`,
                  top: `${currentSelection.range.getBoundingClientRect().bottom + window.scrollY}px`,
                },
              }}
            >
              <Menu.Target>
                <Button size="xs" style={{ visibility: 'hidden' }}>Select as Key</Button>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Item onClick={() => handleSelection()}>Select as Key {selections.length + 1}</Menu.Item>
              </Menu.Dropdown>
            </Menu>
          )}
        </>
      )}

      <Group justify="flex-end" mt="md">
        <Button onClick={onClose}>Close</Button>
      </Group>
    </div>
  );
};

export default InteractiveRecordModal;
