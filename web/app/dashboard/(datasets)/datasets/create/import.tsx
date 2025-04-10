'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { z } from 'zod';
import { useForm, zodResolver } from '@mantine/form';
import classes from '@styles/global.module.css';
import { SelectOptionComponent } from '@/components/SelectOption/SelectOptionComponent';
import {
  TextInput,
  Paper,
  Title,
  Text,
  Container,
  Group,
  Button,
  Select,
  NumberInput,
  Loader,
  Box,
  SimpleGrid,
  Divider,
  MultiSelect,
  Checkbox,
  Alert,
} from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { API_BASE_URL, API_ENDPOINTS } from '@/app/config/api';

interface FormValues {
  dataSource: string;
  datasetName: string;
  description: string;

  // HuggingFace specific
  hfDatasetName: string;
  hfConfig: string;
  hfSplit: string;
  hfToken: string;
  textField: string;
  labelField: string;
  limitRows: number | undefined;
  hfRevision: string;
  selectedColumns: string[];
}

const schema = z.object({
  dataSource: z.string().min(1, { message: 'Data source is required' }),
  datasetName: z.string().min(1, { message: 'Dataset name is required' }),
  description: z.string().optional(),

  // HuggingFace specific
  hfDatasetName: z.string().optional(),
  hfConfig: z.string().optional(),
  hfSplit: z.string().optional(),
  hfToken: z.string().optional(),
  textField: z.string().optional(),
  labelField: z.string().optional(),
  limitRows: z.number().optional(),
  hfRevision: z.string().optional(),
  selectedColumns: z.array(z.string()).optional(),
});

export default function AuthenticationTitle({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State for dynamic form options
  const [configs, setConfigs] = useState<string[]>([]);
  const [splits, setSplits] = useState<string[]>([]);
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [allColumnsSelected, setAllColumnsSelected] = useState(true);
  const [isLoadingConfigs, setIsLoadingConfigs] = useState(false);
  const [isLoadingSplits, setIsLoadingSplits] = useState(false);
  const [isLoadingFeatures, setIsLoadingFeatures] = useState(false);

  const form = useForm<FormValues>({
    validate: zodResolver(schema),
    initialValues: {
      dataSource: 'HuggingFace',
      datasetName: '',
      description: '',

      // HuggingFace specific
      hfDatasetName: '',
      hfConfig: '',
      hfSplit: '',
      hfToken: '',
      textField: '',
      labelField: '',
      limitRows: undefined,
      hfRevision: '',
      selectedColumns: [],
    },
  });

  // Load configs when HuggingFace dataset name changes
  useEffect(() => {
    const fetchConfigs = async () => {
      const hfDatasetName = form.values.hfDatasetName;
      if (!hfDatasetName) {
        setConfigs([]);
        return;
      }

      setIsLoadingConfigs(true);
      try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.datasets.huggingface.configs}?dataset_name=${encodeURIComponent(hfDatasetName)}`);
        if (response.ok) {
          const data = await response.json();
          setConfigs(data.configs || []);
        } else {
          console.error('Failed to fetch configs');
          setConfigs([]);
        }
      } catch (error) {
        console.error('Error fetching configs:', error);
        setConfigs([]);
      } finally {
        setIsLoadingConfigs(false);
      }
    };

    fetchConfigs();
  }, [form.values.hfDatasetName]);

  // Load splits when config changes
  useEffect(() => {
    const fetchSplits = async () => {
      const { hfDatasetName, hfConfig } = form.values;
      if (!hfDatasetName) {
        setSplits([]);
        return;
      }

      setIsLoadingSplits(true);
      try {
        let splitsUrl = `${API_BASE_URL}${API_ENDPOINTS.datasets.huggingface.splits}?dataset_name=${encodeURIComponent(hfDatasetName)}`;
        if (hfConfig) splitsUrl += `&config=${encodeURIComponent(hfConfig)}`;

        const response = await fetch(splitsUrl);
        if (response.ok) {
          const data = await response.json();
          setSplits(data.splits || []);
        } else {
          console.error('Failed to fetch splits');
          setSplits([]);
        }
      } catch (error) {
        console.error('Error fetching splits:', error);
        setSplits([]);
      } finally {
        setIsLoadingSplits(false);
      }
    };

    fetchSplits();
  }, [form.values.hfDatasetName, form.values.hfConfig]);

  // Load features when dataset and config change
  useEffect(() => {
    const fetchFeatures = async () => {
      const { hfDatasetName, hfConfig, hfToken } = form.values;
      if (!hfDatasetName) {
        setFeatures({});
        form.setFieldValue('selectedColumns', []);
        return;
      }

      setIsLoadingFeatures(true);
      try {
        let featuresUrl = `${API_BASE_URL}${API_ENDPOINTS.datasets.huggingface.features}?dataset_name=${encodeURIComponent(hfDatasetName)}`;
        if (hfConfig) featuresUrl += `&config=${encodeURIComponent(hfConfig)}`;
        if (hfToken) featuresUrl += `&token=${encodeURIComponent(hfToken)}`;

        const response = await fetch(featuresUrl);
        if (response.ok) {
          const data = await response.json();
          const columnsData = data.columns || {};
          setFeatures(columnsData);
          if (allColumnsSelected) {
            form.setFieldValue('selectedColumns', Object.keys(columnsData));
          }
        } else {
          console.error('Failed to fetch features');
          setFeatures({});
          form.setFieldValue('selectedColumns', []);
          notifications.show({
            title: 'Could Not Fetch Columns',
            message: 'Failed to retrieve column data for the selected dataset/configuration.',
            color: 'orange',
            icon: <IconInfoCircle />,
          });
        }
      } catch (error) {
        console.error('Error fetching features:', error);
        setFeatures({});
        form.setFieldValue('selectedColumns', []);
        notifications.show({
          title: 'Error Fetching Columns',
          message: 'An error occurred while trying to fetch column data.',
          color: 'red',
          icon: <IconInfoCircle />,
        });
      } finally {
        setIsLoadingFeatures(false);
      }
    };

    fetchFeatures();
  }, [form.values.hfDatasetName, form.values.hfConfig, form.values.hfToken, allColumnsSelected]);

  const handleSelectAllChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const checked = event.currentTarget.checked;
    setAllColumnsSelected(checked);
    if (checked) {
      form.setFieldValue('selectedColumns', Object.keys(features));
    } else {
      form.setFieldValue('selectedColumns', []);
    }
  };

  const handleSubmit = async (values: FormValues) => {
    setIsLoading(true);
    setError(null);

    try {
      let requestBody: any = {
        name: values.datasetName,
        description: values.description,
        source: values.dataSource.toLowerCase(),
      };

      if (values.dataSource === 'HuggingFace') {
        requestBody = {
          ...requestBody,
          identifier: values.hfDatasetName,
          hf_dataset_name: values.hfDatasetName,
          hf_config: values.hfConfig || undefined,
          hf_split: values.hfSplit || undefined,
          hf_revision: values.hfRevision || undefined,
          hf_token: values.hfToken || undefined,
          text_field: values.textField || undefined,
          label_field: values.labelField || undefined,
          selected_columns: allColumnsSelected ? undefined : values.selectedColumns,
          limit_rows: values.limitRows,
        };

        if (!requestBody.hf_dataset_name) throw new Error("HuggingFace Dataset ID is required.");
        if (!requestBody.hf_split) throw new Error("HuggingFace Split is required.");
        if (!requestBody.text_field) throw new Error("Text Field is required. Please specify which column contains the text content.");
        if (!allColumnsSelected && (!values.selectedColumns || values.selectedColumns.length === 0)) {
          throw new Error("Please select at least one column to import.");
        }
        if (!allColumnsSelected && values.textField && !values.selectedColumns.includes(values.textField)) {
          throw new Error("The specified Text Field must be included in the selected columns.");
        }
        if (!allColumnsSelected && values.labelField && !values.selectedColumns.includes(values.labelField)) {
          throw new Error("The specified Label Field must be included in the selected columns.");
        }
      }

      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.datasets.base}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create dataset');
      }

      const result = await response.json();
      console.log('Dataset created:', result);
      router.push('/dashboard/datasets');
    } catch (error: any) {
      console.error('Failed to create dataset:', error);
      setError(error.message || 'Failed to create dataset. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Determine if we should show HuggingFace specific fields
  const showHuggingFaceFields = form.values.dataSource === 'HuggingFace';

  // Prepare column data for MultiSelect
  const columnSelectData = Object.entries(features).map(([name, type]) => ({
    value: name,
    label: `${name} (${type})`
  }));

  return (
    <div>
      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Container size={700} my={40}>
          <Title ta="center" className={classes.h1}>
            Create a Dataset
          </Title>

          <Text size="sm" ta="center" mt={5} mb={20}>
            Import data from Hugging Face and other sources
          </Text>

          <Paper withBorder shadow="md" p={30} radius="md">
            <SelectOptionComponent
              value={form.values.dataSource}
              onChange={(value) => form.setFieldValue('dataSource', value)}
              disabledOptions={['CSV', 'Parquet']}
            />

            <Divider my="md" />

            <SimpleGrid cols={2} spacing="md" verticalSpacing="md">
              <TextInput
                {...form.getInputProps('datasetName')}
                label="Dataset Name"
                placeholder="Enter name for your dataset"
                withAsterisk
              />

              <TextInput
                {...form.getInputProps('description')}
                label="Description"
                placeholder="Brief description of the dataset"
              />
            </SimpleGrid>

            {form.values.dataSource === 'HuggingFace' && (
              <>
                <Divider my="md" label="Hugging Face Settings" labelPosition="center" />

                <SimpleGrid cols={2} spacing="md" verticalSpacing="md">
                  <TextInput
                    {...form.getInputProps('hfDatasetName')}
                    label="HF Dataset ID"
                    placeholder="e.g., databricks/dolly-15k"
                    withAsterisk
                  />

                  <Select
                    {...form.getInputProps('hfConfig')}
                    label="Configuration"
                    placeholder={isLoadingConfigs ? "Loading..." : "Select a configuration"}
                    data={configs.map(config => ({ value: config, label: config }))}
                    disabled={isLoadingConfigs || configs.length === 0}
                    clearable
                    rightSection={isLoadingConfigs ? <Loader size="xs" /> : null}
                  />
                </SimpleGrid>

                <TextInput
                  {...form.getInputProps('hfRevision')}
                  label="Revision / Commit Hash (Optional)"
                  placeholder="Defaults to main branch"
                  mt="md"
                />

                <SimpleGrid cols={2} spacing="md" verticalSpacing="md" mt="md">
                  <Select
                    {...form.getInputProps('hfSplit')}
                    label="Split"
                    placeholder={isLoadingSplits ? "Loading..." : "Select a split"}
                    data={splits.map(split => ({ value: split, label: split }))}
                    disabled={isLoadingSplits || splits.length === 0}
                    clearable
                    rightSection={isLoadingSplits ? <Loader size="xs" /> : null}
                  />

                  <TextInput
                    {...form.getInputProps('hfToken')}
                    label="API Token (for private datasets)"
                    placeholder="Optional"
                  />
                </SimpleGrid>

                <Divider my="lg" label="Column Selection" labelPosition="center" />
                {isLoadingFeatures ? (
                  <Group justify="center"><Loader size="sm" /> <Text size="sm">Loading columns...</Text></Group>
                ) : !form.values.hfDatasetName ? (
                  <Text size="sm" c="dimmed" ta="center">Enter a dataset name above to view available columns</Text>
                ) : columnSelectData.length > 0 ? (
                  <>
                    <Checkbox
                      label="Import All Columns"
                      checked={allColumnsSelected}
                      onChange={handleSelectAllChange}
                      mb="sm"
                    />
                    <MultiSelect
                      label="Columns to Import"
                      placeholder="Select columns"
                      data={columnSelectData}
                      {...form.getInputProps('selectedColumns')}
                      disabled={allColumnsSelected || isLoadingFeatures}
                      searchable
                      clearable
                    />
                  </>
                ) : (
                  <Text size="sm" c="dimmed" ta="center">
                    No columns found for the specified dataset/configuration, or the fetch failed.
                  </Text>
                )}

                <Divider my="lg" label="Field Mapping & Limit" labelPosition="center" />

                <SimpleGrid cols={2} spacing="md" verticalSpacing="md">
                  <TextInput
                    {...form.getInputProps('textField')}
                    label="Text Field"
                    placeholder="Column containing main text content"
                    description={form.values.textField && !form.values.selectedColumns.includes(form.values.textField) && !allColumnsSelected ? "Warning: Text Field not selected above" : ""}
                    withAsterisk
                  />

                  <TextInput
                    {...form.getInputProps('labelField')}
                    label="Label Field"
                    placeholder="Optional column containing labels"
                    description={form.values.labelField && !form.values.selectedColumns.includes(form.values.labelField) && !allColumnsSelected ? "Warning: Label Field not selected above" : ""}
                  />
                </SimpleGrid>

                <Box mt="md">
                  <NumberInput
                    {...form.getInputProps('limitRows')}
                    label="Row Limit (Optional)"
                    description="Maximum rows to download (leave empty for all rows)"
                    min={1}
                    placeholder="Download all rows"
                    w="100%"
                  />
                </Box>
              </>
            )}

            <Button fullWidth mt={30} type="submit" loading={isLoading}>
              Create Dataset
            </Button>

            {error && <Text c="red" mt="sm">{error}</Text>}
          </Paper>
        </Container>
      </form>
    </div>
  );
}