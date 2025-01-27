'use client';
import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  Text,
  rem,
  Divider,
  Button,
  Group,
  Switch,
  useMantineTheme,
  Badge,
  ActionIcon,
  Highlight,
  CopyButton,
  Tooltip,
  Modal,
} from '@mantine/core';
import {
  IconCopy,
  IconCheck,
  IconX,
  IconChevronLeft,
  IconChevronRight,
  IconMaximize,
} from '@tabler/icons-react';
import ReactMarkdown from 'react-markdown';

import InteractiveRecordModal from './expand';
import { AccordionStats } from './(components)/accordion';
import { SortButton } from './sort';
import SearchComponent from './search';
import CustomPagination from './pagination';

interface Record {
  [key: string]: string;
}

interface ApiResponse {
  split: string;
  total_records: number;
  page: number;
  page_size: number;
  records: Record[];
}

// Our selection interface
interface Selection {
  type: 'key' | 'value';
  number: number;
  text: string;
  range: Range;
}

const LeadGrid: React.FC = () => {
  const theme = useMantineTheme();
  const [renderMarkdown, setRenderMarkdown] = useState<boolean>(false);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalRecords, setTotalRecords] = useState<number>(0);
  const [allRecords, setAllRecords] = useState<Record[]>([]);
  const [filteredRecords, setFilteredRecords] = useState<Record[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerms, setSearchTerms] = useState<string[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<{ key: string; value: string } | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    filterRecords();
  }, [searchTerms, allRecords]);

  const fetchData = async () => {
    setLoading(true);
    try {
      let allFetchedRecords: Record[] = [];
      let page = 1;
      let total = 0;
      const pageSize = 50;

      while (true) {
        const response = await fetch(
          `http://127.0.0.1:8000/datasets/29/records?page=${page}&page_size=${pageSize}`
        );
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data: ApiResponse = await response.json();
        total = data.total_records;

        allFetchedRecords = [...allFetchedRecords, ...data.records];

        if (allFetchedRecords.length >= total) {
          break;
        }
        page += 1;
      }

      setAllRecords(allFetchedRecords);
      setFilteredRecords(allFetchedRecords); // Start with the same as allRecords
      setTotalRecords(total);
      setError(null);
    } catch (error: any) {
      setError('Failed to fetch data');
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const filterRecords = () => {
    if (searchTerms.length === 0) {
      setFilteredRecords(allRecords);
    } else {
      const filtered = allRecords.filter((record) =>
        Object.values(record).some((value) =>
          searchTerms.some((term) => value.toLowerCase().includes(term.toLowerCase()))
        )
      );
      setFilteredRecords(filtered);
    }
    setCurrentPage(1);
  };

  const handleSearch = (term: string) => {
    const terms = term.split(' ').filter((t) => t.trim() !== '');
    setSearchTerms(terms);
  };

  /**
   * Called by the InteractiveRecordModal after the user highlights text
   * and chooses "Key" or "Value," then hits "Apply."
   */
  const handleApplySelections = (selections: Selection[]) => {
    setModalOpen(false);
    if (!selectedRecord) return;

    // Get all key selections in order
    const keys = selections
      .filter(s => s.type === 'key')
      .sort((a, b) => a.range.startOffset - b.range.startOffset);

    if (keys.length === 0) return;

    // Update both allRecords and filteredRecords
    const updateRecords = (records: Record[]) => {
      return records.map(record => {
        // Only process records that have a chat field
        if (!record.chat) {
          return record;
        }

        const newRecord = { ...record };

        // Remove any existing structured fields
        Object.keys(newRecord).forEach(key => {
          if (key.startsWith('structured_key_') || key.startsWith('structured_value_')) {
            delete newRecord[key];
          }
        });

        // Split the chat content by the selected keys
        const chatContent = record.chat;
        let structuredIndex = 1;

        // Find all occurrences of each key
        const allOccurrences: { key: string; index: number }[] = [];

        // For each key type (e.g., "USER:", "A:")
        keys.forEach(keySelection => {
          let pos = 0;
          const keyText = keySelection.text;

          // Find all occurrences of this key
          while (pos < chatContent.length) {
            const index = chatContent.indexOf(keyText, pos);
            if (index === -1) break;

            allOccurrences.push({
              key: keyText,
              index: index
            });
            pos = index + keyText.length;
          }
        });

        // Sort all occurrences by their position in the text
        allOccurrences.sort((a, b) => a.index - b.index);

        // Process each occurrence
        allOccurrences.forEach((occurrence, index) => {
          const startIndex = occurrence.index + occurrence.key.length;
          const nextOccurrence = allOccurrences[index + 1];
          const endIndex = nextOccurrence ? nextOccurrence.index : chatContent.length;

          // Extract the value (everything between this key and the next)
          const value = chatContent
            .slice(startIndex, endIndex)
            .trim();

          // Add the structured fields
          newRecord[`structured_key_${structuredIndex}`] = occurrence.key;
          newRecord[`structured_value_${structuredIndex}`] = value;

          structuredIndex++;
        });

        return newRecord;
      });
    };

    setAllRecords(prev => {
      const updated = updateRecords(prev);
      console.log('All records updated:', updated);
      return updated;
    });

    setFilteredRecords(prev => {
      const updated = updateRecords(prev);
      console.log('Filtered records updated:', updated);
      return updated;
    });
  };

  const currentRecord = filteredRecords[currentPage - 1] || null;
  const totalPages = filteredRecords.length;

  const preStyles: React.CSSProperties = {
    fontFamily: 'inherit',
    fontWeight: 300,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    width: '100%',
    margin: 0,
    padding: theme.spacing.sm,
    backgroundColor: 'lavenderblush',
    borderRadius: theme.radius.md,
    border: '1px solid black',
  };

  const customRenderers = {
    p: ({ node, children }: any) => {
      const content = Array.isArray(children) ? children.join('') : children;
      return <Highlight highlight={searchTerms}>{content}</Highlight>;
    },
  };

  // Add this helper function at component level
  const getStructuredFields = (record: Record) => {
    const fields: { key: string; value: string }[] = [];
    let index = 1;

    while (record[`structured_key_${index}`]) {
      fields.push({
        key: record[`structured_key_${index}`],
        value: record[`structured_value_${index}`]
      });
      index++;
    }

    return fields;
  };

  // Helper to get non-structured fields
  const getOriginalFields = (record: Record) => {
    return Object.entries(record).filter(([key]) => 
      !key.startsWith('structured_key_') && 
      !key.startsWith('structured_value_')
    );
  };

  return (
    <>
      <Group mt={10} position="apart">
        <SearchComponent onSearch={handleSearch} />
        <Group spacing="xs">
          <ActionIcon color="gray">
            <IconChevronLeft size="1rem" />
          </ActionIcon>
          <Button variant="default">Filter</Button>
          <SortButton />
        </Group>
      </Group>

      <Container my="md" fluid>
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 8 }}>
            <Card shadow="sm" radius="md" withBorder pb={50}>
              <Card.Section
                withBorder
                inheritPadding
                py="xs"
                style={{ backgroundColor: 'lavenderblush' }}
              >
                <Group position="apart" align="center">
                  <Text fw={300} size="sm">
                    Dataset Record
                  </Text>
                  <CustomPagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    onPageChange={setCurrentPage}
                  />
                  <Switch
                    checked={renderMarkdown}
                    onChange={(event) => setRenderMarkdown(event.currentTarget.checked)}
                    color="teal"
                    size="sm"
                    label="Markdown"
                    thumbIcon={
                      renderMarkdown ? (
                        <IconCheck
                          style={{ width: rem(12), height: rem(12) }}
                          color={theme.colors.teal[6]}
                          stroke={3}
                        />
                      ) : (
                        <IconX
                          style={{ width: rem(12), height: rem(12) }}
                          color={theme.colors.red[6]}
                          stroke={3}
                        />
                      )
                    }
                  />
                </Group>
              </Card.Section>

              <Card.Section inheritPadding mt="md">
                {loading ? (
                  <Text>Loading...</Text>
                ) : error ? (
                  <Text c="red">{error}</Text>
                ) : currentRecord ? (
                  <>
                    {/* Show all original fields */}
                    {getOriginalFields(currentRecord).map(([fieldKey, fieldValue], index) => (
                      <React.Fragment key={fieldKey}>
                        <Group p="apart" mb="xs">
                          <Badge color={fieldKey === 'chat' ? 'blue' : 'gray'}>{fieldKey}</Badge>
                          <Group justify="xs">
                            <CopyButton value={fieldValue} timeout={2000}>
                              {({ copied, copy }) => (
                                <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow position="right">
                                  <ActionIcon color={copied ? 'teal' : 'gray'} variant="subtle" onClick={copy}>
                                    {copied ? (
                                      <IconCheck style={{ width: rem(16) }} />
                                    ) : (
                                      <IconCopy style={{ width: rem(16) }} />
                                    )}
                                  </ActionIcon>
                                </Tooltip>
                              )}
                            </CopyButton>
                            <Tooltip label="Expand" withArrow position="right">
                              <ActionIcon
                                color="blue"
                                variant="subtle"
                                onClick={() => {
                                  setSelectedRecord({ key: fieldKey, value: fieldValue });
                                  setModalOpen(true);
                                }}
                              >
                                <IconMaximize style={{ width: rem(16) }} />
                              </ActionIcon>
                            </Tooltip>
                          </Group>
                        </Group>
                        <pre style={preStyles}>
                          <Highlight highlight={searchTerms}>{fieldValue}</Highlight>
                        </pre>
                        {index < getOriginalFields(currentRecord).length - 1 && <Divider my="md" />}
                      </React.Fragment>
                    ))}

                    {/* Show structured fields as branches */}
                    {getStructuredFields(currentRecord).length > 0 && (
                      <>
                        <Divider my="md" label="Structured Content" labelPosition="center" />
                        <div style={{ paddingLeft: rem(20) }}>
                          {getStructuredFields(currentRecord).map((field, index) => (
                            <React.Fragment key={index}>
                              <Group spacing="xs" mb="xs">
                                <Group spacing={4}>
                                  {/* Tree-like structure */}
                                  <Text color="dimmed" size="sm">└─</Text>
                                  <Badge 
                                    color="pink"
                                    variant="light"
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => {
                                      setSelectedRecord({ key: field.key, value: field.value });
                                      setModalOpen(true);
                                    }}
                                  >
                                    {field.key}
                                  </Badge>
                                </Group>
                                <Group justify="xs">
                                  <CopyButton value={field.value} timeout={2000}>
                                    {({ copied, copy }) => (
                                      <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow position="right">
                                        <ActionIcon color={copied ? 'teal' : 'gray'} variant="subtle" onClick={copy}>
                                          {copied ? (
                                            <IconCheck style={{ width: rem(16) }} />
                                          ) : (
                                            <IconCopy style={{ width: rem(16) }} />
                                          )}
                                        </ActionIcon>
                                      </Tooltip>
                                    )}
                                  </CopyButton>
                                  <Tooltip label="Expand" withArrow position="right">
                                    <ActionIcon
                                      color="blue"
                                      variant="subtle"
                                      onClick={() => {
                                        setSelectedRecord({ key: field.key, value: field.value });
                                        setModalOpen(true);
                                      }}
                                    >
                                      <IconMaximize style={{ width: rem(16) }} />
                                    </ActionIcon>
                                  </Tooltip>
                                </Group>
                              </Group>
                              <div style={{ paddingLeft: rem(35) }}>
                                {renderMarkdown ? (
                                  <ReactMarkdown components={customRenderers}>{field.value}</ReactMarkdown>
                                ) : (
                                  <pre style={preStyles}>
                                    <Highlight highlight={searchTerms}>{field.value}</Highlight>
                                  </pre>
                                )}
                              </div>
                              {index < getStructuredFields(currentRecord).length - 1 && 
                                <Divider my="md" variant="dashed" />}
                            </React.Fragment>
                          ))}
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  <Text>No record found</Text>
                )}
              </Card.Section>
            </Card>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 4 }}>
            <AccordionStats />
          </Grid.Col>
        </Grid>

        <Modal
          opened={modalOpen}
          onClose={() => setModalOpen(false)}
          centered
          title={selectedRecord?.key}
          size="lg"
        >
          {selectedRecord && (
            <InteractiveRecordModal
              record={selectedRecord}
              onApplySelections={handleApplySelections}
            />
          )}
        </Modal>
      </Container>
    </>
  );
};

export default LeadGrid;
