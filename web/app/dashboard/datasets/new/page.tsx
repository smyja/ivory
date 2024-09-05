'use client'
import React, { useState, useEffect } from 'react';
import { Container, Grid, Card, Text, rem, Divider, Button, Group, Switch, useMantineTheme, Badge, ActionIcon, Highlight, CopyButton, Tooltip } from '@mantine/core';
import ReactMarkdown from 'react-markdown';
import { IconCopy, IconCheck, IconX, IconChevronLeft, IconChevronRight } from '@tabler/icons-react';
import { AccordionStats } from '../accordion';
import { SortButton } from './sort';
import SearchComponent from './search';

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

interface CustomPaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const CustomPagination: React.FC<CustomPaginationProps> = ({ currentPage, totalPages, onPageChange }) => {
  return (
    <Group gap={5} align="center">
      <ActionIcon 
        color='gray'
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={currentPage === 1}
      >
        <IconChevronLeft size="1rem" />
      </ActionIcon>
      <Text size="sm">
        {currentPage} of {totalPages.toLocaleString()}
      </Text>
      <ActionIcon 
        color='gray'
        onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        disabled={currentPage === totalPages}
      >
        <IconChevronRight size="1rem" />
      </ActionIcon>
    </Group>
  );
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
          let totalRecords = 0;
          const pageSize = 50; // Assuming the backend is returning 50 records per page
      
          while (true) {
            const response = await fetch(`http://0.0.0.0:8000/datasets/7/records?page=${page}&page_size=${pageSize}`);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data: ApiResponse = await response.json();
      totalRecords = data.total_records;
      
            // Combine records from each page
            allFetchedRecords = [...allFetchedRecords, ...data.records];
      
            // If we've fetched all records, break the loop
            if (allFetchedRecords.length >= totalRecords) {
              break;
            }
      
            page += 1; // Move to the next page
          }
      
          setAllRecords(allFetchedRecords);
      setTotalRecords(totalRecords);
      setError(null);
    } catch (error) {
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
      const filtered = allRecords.filter(record =>
        Object.values(record).some(value =>
          searchTerms.some(term =>
            value.toLowerCase().includes(term.toLowerCase())
          )
        )
      );
      setFilteredRecords(filtered);
    }
    setCurrentPage(1);
  };

  const handleSearch = (term: string) => {
    const terms = term.split(' ').filter(t => t.trim() !== '');
    setSearchTerms(terms);
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

  return (
    <>
      <Group mt={10} justify="space-between">
        <SearchComponent onSearch={handleSearch} />
        <Group justify="flex-end" gap="xs">
          <ActionIcon color='gray'>
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
              <Card.Section withBorder inheritPadding py="xs" style={{ backgroundColor: "lavenderblush" }}>
                <Group justify="space-between" align="center">
                  <Text fw={300} size="sm">Dataset Record</Text>
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
                  Object.entries(currentRecord).map(([key, value], index) => (
                    <React.Fragment key={index}>
                      <Group justify="space-between" mb="xs">
                        <Badge color="rgba(255, 110, 110, 1)">{key}</Badge>
                        <CopyButton value={value} timeout={2000}>
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
                      </Group>
                      {renderMarkdown ? (
                        <ReactMarkdown components={customRenderers}>{value}</ReactMarkdown>
                      ) : (
                        <pre style={preStyles}>
                          <Highlight highlight={searchTerms}>{value}</Highlight>
                        </pre>
                      )}
                      {index < Object.entries(currentRecord).length - 1 && <Divider my="md" />}
                    </React.Fragment>
                  ))
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
      </Container>
    </>
  );
};

export default LeadGrid;
