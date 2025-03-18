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

const DatasetView: React.FC = () => {
  const theme = useMantineTheme();
  const searchParams = useSearchParams();
  const router = useRouter();
  const datasetId = searchParams.get('id');

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

  useEffect(() => {
    if (!datasetId) {
      notifications.show({
        title: 'Error',
        message: 'No dataset ID provided',
        color: 'red',
      });
      router.push('/dashboard/datasets');
      return;
    }
    fetchDatasetInfo();
  }, [datasetId]);

  const fetchDatasetInfo = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch dataset info');
      }
      const data = await response.json();
      setDatasetInfo(data);
      fetchData();
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

  const fetchData = async () => {
    if (!datasetId) return;

    setLoading(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/records?page=${currentPage}&page_size=10`
      );
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
        throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      setAllRecords(data.records);
      setFilteredRecords(data.records);
      setTotalRecords(data.total_records);
      setError(null);
    } catch (error: any) {
      setError(error.message || 'Failed to fetch data');
      console.error('Error fetching data:', error);
      notifications.show({
        title: 'Error',
        message: error.message || 'Failed to fetch dataset records',
        color: 'red',
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [currentPage]);

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
      <LoadingOverlay visible={loading} overlayProps={{ blur: 2 }} />
      <Card shadow="sm" radius="md" withBorder mb="md">
        <Group justify="space-between" mb="md">
          <div>
            <Title order={2}>{datasetInfo?.name || 'Loading...'}</Title>
            <Text c="dimmed" size="sm">
              {datasetInfo?.subset ? `Subset: ${datasetInfo.subset}` : 'No subset'} •{' '}
              {datasetInfo?.split ? `Split: ${datasetInfo.split}` : 'No split'}
            </Text>
          </div>
          <Button variant="light" onClick={() => router.push('/dashboard/datasets')}>
            Back to Datasets
          </Button>
        </Group>
      </Card>

      <Grid gutter="md">
        <Grid.Col span={{ base: 12, md: 8 }}>
          <Card shadow="sm" radius="md" withBorder pb={50}>
            <Card.Section
              withBorder
              inheritPadding
              py="xs"
              style={{ backgroundColor: 'lavenderblush' }}
            >
              <Group justify="space-between" align="center">
                <Text fw={300} size="sm">
                  Dataset Record
                </Text>
                <Group gap="xs">
                  <CustomPagination
                    currentPage={currentPage}
                    totalPages={Math.ceil(totalRecords / 10)}
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
