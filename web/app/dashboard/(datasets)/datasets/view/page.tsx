'use client';
import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
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
  Title,
  LoadingOverlay,
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
import { notifications } from '@mantine/notifications';
import { useRouter } from 'next/navigation';

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

interface DatasetInfo {
  id: number;
  name: string;
  subset: string | null;
  split: string | null;
  status: string;
  splits: {
    [key: string]: {
      columns: string[];
      row_count: number;
      sample_rows: Record[];
    };
  };
}

const BACKEND_PAGE_SIZE = 10; // Fetch 100 records at a time from backend

const DatasetView: React.FC = () => {
  const theme = useMantineTheme();
  const searchParams = useSearchParams();
  const router = useRouter();
  const datasetId = searchParams.get('id');
  const subclusterId = searchParams.get('subcluster');

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
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [records, setRecords] = useState<Record[]>([]);
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(true);
  const [isPageLoading, setIsPageLoading] = useState<boolean>(false);
  const [currentBackendPage, setCurrentBackendPage] = useState<number>(1);
  const [cachedRecords, setCachedRecords] = useState<Record[]>([]);

  useEffect(() => {
    if (datasetId) {
      fetchDatasetInfo();
      if (subclusterId) {
        fetchSubclusterTexts();
      } else {
        fetchInitialRecords();
      }
    }
  }, [datasetId, subclusterId]);

  // Calculate which backend page we need based on the current frontend page
  useEffect(() => {
    const requiredBackendPage = Math.ceil(currentPage / BACKEND_PAGE_SIZE);
    if (requiredBackendPage !== currentBackendPage) {
      setCurrentBackendPage(requiredBackendPage);
    }
  }, [currentPage]);

  // Fetch backend page when it changes
  useEffect(() => {
    if (datasetId && !subclusterId && !isInitialLoading) {
      fetchBackendPage(currentBackendPage);
    }
  }, [currentBackendPage]);

  // Update displayed record when page changes
  useEffect(() => {
    if (cachedRecords.length > 0) {
      const indexInBackendPage = (currentPage - 1) % BACKEND_PAGE_SIZE;
      if (cachedRecords[indexInBackendPage]) {
        setFilteredRecords([cachedRecords[indexInBackendPage]]);
        setRecords([cachedRecords[indexInBackendPage]]);
      }
    }
  }, [currentPage, cachedRecords]);

  const fetchDatasetInfo = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}?detail_level=full`);
      if (!response.ok) {
        throw new Error('Failed to fetch dataset info');
      }
      const data = await response.json();
      setDatasetInfo(data);
    } catch (error) {
      console.error('Error fetching dataset info:', error);
      notifications.show({
        title: 'Error',
        message: 'Failed to fetch dataset information',
        color: 'red',
      });
      setLoading(false);
    }
  };

  const fetchInitialRecords = async () => {
    try {
      setIsInitialLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}?detail_level=data&page=1&page_size=${BACKEND_PAGE_SIZE}`
      );

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to fetch records';

        switch (response.status) {
          case 404:
            errorMessage = `Dataset not found: ${errorData.detail}`;
            break;
          case 400:
            errorMessage = `Invalid request: ${errorData.detail}`;
            break;
          case 500:
            errorMessage = `Server error: ${errorData.detail}`;
            break;
          default:
            errorMessage = errorData.detail || errorMessage;
        }

        throw new Error(errorMessage);
      }

      const data: ApiResponse = await response.json();
      setCachedRecords(data.records);
      setTotalRecords(data.total_records);
      // Show the first record
      setFilteredRecords([data.records[0]]);
      setRecords([data.records[0]]);
    } catch (error: any) {
      setError(error.message);
      notifications.show({
        title: 'Error',
        message: error.message,
        color: 'red',
      });
    } finally {
      setIsInitialLoading(false);
    }
  };

  const fetchBackendPage = async (backendPage: number) => {
    try {
      setIsPageLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}?detail_level=data&page=${backendPage}&page_size=${BACKEND_PAGE_SIZE}`
      );

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to fetch records';

        switch (response.status) {
          case 404:
            errorMessage = `Dataset not found: ${errorData.detail}`;
            break;
          case 400:
            errorMessage = `Invalid request: ${errorData.detail}`;
            break;
          case 500:
            errorMessage = `Server error: ${errorData.detail}`;
            break;
          default:
            errorMessage = errorData.detail || errorMessage;
        }

        throw new Error(errorMessage);
      }

      const data: ApiResponse = await response.json();
      setCachedRecords(data.records);

      // Calculate the index in the current backend page
      const indexInBackendPage = (currentPage - 1) % BACKEND_PAGE_SIZE;
      setFilteredRecords([data.records[indexInBackendPage]]);
      setRecords([data.records[indexInBackendPage]]);
    } catch (error: any) {
      setError(error.message);
      notifications.show({
        title: 'Error',
        message: error.message,
        color: 'red',
      });
    } finally {
      setIsPageLoading(false);
    }
  };

  const fetchSubclusterTexts = async () => {
    try {
      setIsInitialLoading(true);
      setLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/subclusters/${subclusterId}/texts`
      );

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to fetch subcluster texts';

        switch (response.status) {
          case 404:
            errorMessage = `Subcluster not found: ${errorData.detail}`;
            break;
          case 500:
            errorMessage = `Server error: ${errorData.detail}`;
            break;
          default:
            errorMessage = errorData.detail || errorMessage;
        }

        throw new Error(errorMessage);
      }

      const data = await response.json();
      const formattedRecords = data.texts.map((text: any) => ({
        text: text.text
      }));
      setRecords(formattedRecords);
      setFilteredRecords(formattedRecords);
      setTotalRecords(formattedRecords.length);
      setCachedRecords(formattedRecords);
    } catch (error: any) {
      setError(error.message);
      notifications.show({
        title: 'Error',
        message: error.message,
        color: 'red',
      });
    } finally {
      setIsInitialLoading(false);
      setLoading(false);
    }
  };

  const handleSearch = async (term: string) => {
    const terms = term.split(' ').filter((t) => t.trim() !== '');
    setSearchTerms(terms);

    if (terms.length === 0) {
      fetchInitialRecords();
      return;
    }

    try {
      setIsInitialLoading(true);
      const searchParams = new URLSearchParams({
        page: '1',
        page_size: BACKEND_PAGE_SIZE.toString(),
        search: terms.join(' ')
      });

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/records?${searchParams}`
      );
      if (!response.ok) {
        throw new Error('Failed to search records');
      }
      const data: ApiResponse = await response.json();
      setCachedRecords(data.records);
      setTotalRecords(data.total_records);
      setCurrentPage(1);
      // Show the first record of search results
      setFilteredRecords([data.records[0]]);
    } catch (error: any) {
      notifications.show({
        title: 'Error',
        message: error.message || 'Failed to search records',
        color: 'red',
      });
    } finally {
      setIsInitialLoading(false);
    }
  };

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

  const getOriginalFields = (record: Record) => {
    return Object.entries(record).filter(([key]) =>
      !key.startsWith('structured_key_') &&
      !key.startsWith('structured_value_')
    );
  };

  return (
    <Container my="md" fluid>
      <LoadingOverlay visible={isInitialLoading} overlayProps={{ blur: 2 }} />
      <Card shadow="sm" radius="md" withBorder mb="md">
        <Group justify="space-between" mb="md">
          <div>
            <Title order={2}>{datasetInfo?.name || 'Loading...'}</Title>
            <Text c="dimmed" size="sm">
              {datasetInfo?.subset ? `Subset: ${datasetInfo.subset}` : 'No subset'} •{' '}
              {datasetInfo?.split ? `Split: ${datasetInfo.split}` : 'No split'}
              {subclusterId && ' • Viewing Subcluster Texts'}
            </Text>
          </div>
          <Group>
            {!subclusterId && (
              <Button
                variant="filled"
                color="black"
                radius="md"
                onClick={() => router.push(`/dashboard/datasets/cluster?id=${datasetId}`)}
                styles={{
                  root: {
                    transition: 'background-color 0.2s ease',
                    '&:hover': {
                      backgroundColor: '#333',
                    },
                  },
                }}
              >
                View Clusters
              </Button>
            )}
            <Button
              variant="light"
              color="black"
              radius="md"
              onClick={() => router.push('/dashboard/datasets')}
              styles={{
                root: {
                  transition: 'background-color 0.2s ease',
                  '&:hover': {
                    backgroundColor: '#f0f0f0',
                  },
                },
              }}
            >
              Back to Datasets
            </Button>
          </Group>
        </Group>
      </Card>

      <Grid gutter="md">
        <Grid.Col span={{ base: 12, md: 8 }}>
          <Card shadow="sm" radius="md" withBorder pb={50}>
            <Card.Section
              withBorder
              inheritPadding
              py="xs"
              style={{
                backgroundColor: 'lavenderblush',
                opacity: isPageLoading ? 0.7 : 1,
                transition: 'opacity 0.2s ease'
              }}
            >
              <Group justify="space-between" align="center">
                <Text fw={300} size="sm">
                  Dataset Record {isPageLoading && '(Loading...)'}
                </Text>
                <Group gap="xs">
                  <CustomPagination
                    currentPage={currentPage}
                    totalPages={totalRecords} // Since PAGE_SIZE is 1, totalPages equals totalRecords
                    onPageChange={setCurrentPage}
                    disabled={isPageLoading}
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
              </Group>
            </Card.Section>

            <Card.Section inheritPadding mt="md">
              {error ? (
                <Text c="red">{error}</Text>
              ) : filteredRecords.length > 0 ? (
                <>
                  {getOriginalFields(filteredRecords[0]).map(([fieldKey, fieldValue], index) => (
                    <React.Fragment key={fieldKey}>
                      <Group justify="space-between" mb="xs">
                        <Badge color={fieldKey === 'chat' ? 'blue' : 'gray'}>{fieldKey}</Badge>
                        <Group gap="xs">
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
                      {index < getOriginalFields(filteredRecords[0]).length - 1 && <Divider my="md" />}
                    </React.Fragment>
                  ))}

                  {getStructuredFields(filteredRecords[0]).length > 0 && (
                    <>
                      <Divider my="md" label="Structured Content" labelPosition="center" />
                      <div style={{ paddingLeft: rem(20) }}>
                        {getStructuredFields(filteredRecords[0]).map((field, index) => (
                          <React.Fragment key={index}>
                            <Group gap="xs" mb="xs">
                              <Group gap={4}>
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
                              <Group gap="xs">
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
                            {index < getStructuredFields(filteredRecords[0]).length - 1 &&
                              <Divider my="md" variant="dashed" />}
                          </React.Fragment>
                        ))}
                      </div>
                    </>
                  )}
                </>
              ) : (
                <Text>No records found</Text>
              )}
            </Card.Section>
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 4 }}>
          <AccordionStats />
        </Grid.Col>
      </Grid>

      {selectedRecord && (
        <Modal
          opened={modalOpen}
          onClose={() => setModalOpen(false)}
          size="xl"
          title={selectedRecord.key}
        >
          <InteractiveRecordModal
            content={selectedRecord.value}
            onClose={() => setModalOpen(false)}
          />
        </Modal>
      )}
    </Container>
  );
};

export default DatasetView;
