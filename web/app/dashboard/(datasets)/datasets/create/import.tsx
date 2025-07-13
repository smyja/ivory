'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { z } from 'zod';
import { useForm, zodResolver } from '@mantine/form';
import classes from '@styles/global.module.css';
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
  Progress,
} from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { SelectOptionComponent } from '@/components/SelectOption/SelectOptionComponent';
import { API_BASE_URL, API_ENDPOINTS } from '@/app/config/api';
import { useDownloads } from '@/components/DownloadNotifications/DownloadContext';

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
  textFields: string[];
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
  textFields: z
    .array(z.string())
    .min(1, { message: 'Please select at least one Text Field for clustering' })
    .optional(),
});

export default function AuthenticationTitle({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { addDownload } = useDownloads();

  // State for dynamic form options
  const [configs, setConfigs] = useState<string[]>([]);
  const [splits, setSplits] = useState<string[]>([]);
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [allColumnsSelected, setAllColumnsSelected] = useState(true);
  const [isLoadingConfigs, setIsLoadingConfigs] = useState(false);
  const [isLoadingSplits, setIsLoadingSplits] = useState(false);
  const [isLoadingFeatures, setIsLoadingFeatures] = useState(false);

  const [isSubmitting, setIsSubmitting] = useState(false);

  const [featureOptions, setFeatureOptions] = useState<{ value: string; label: string }[]>([]);

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
      textFields: [],
    },
  });

  // Load configs when HuggingFace dataset name changes
  useEffect(() => {
    const fetchConfigs = async () => {
      const { hfDatasetName } = form.values;
      if (!hfDatasetName) {
        setConfigs([]);
        return;
      }

      setIsLoadingConfigs(true);
      try {
        const response = await fetch(
          `${API_BASE_URL}${API_ENDPOINTS.datasets.huggingface.configs}?dataset_name=${encodeURIComponent(hfDatasetName)}`
        );
        if (response.ok) {
          const data = await response.json();

          let configsArray: string[] = [];

          if (Array.isArray(data)) {
            // Case 1: API returned an array directly
            configsArray = data;
          } else if (data.configs && Array.isArray(data.configs)) {
            // Case 2: API returned object with 'configs' property
            configsArray = data.configs;
          }

          setConfigs(configsArray);
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

        console.log('Fetching splits from:', splitsUrl);
        const response = await fetch(splitsUrl);

        if (response.ok) {
          const data = await response.json();
          console.log('Splits API response:', data);

          // Handle both response formats:
          // 1. Array directly (your API response)
          // 2. Object with a 'splits' property containing an array (expected format)
          let splitsArray: string[] = [];

          if (Array.isArray(data)) {
            // Case 1: API returned an array directly
            splitsArray = data;
            console.log('API returned direct array of splits:', splitsArray);
          } else if (data.splits && Array.isArray(data.splits)) {
            // Case 2: API returned object with 'splits' property
            splitsArray = data.splits;
            console.log('API returned splits property:', splitsArray);
          }

          if (splitsArray.length > 0) {
            console.log('Setting splits to:', splitsArray);
            setSplits(splitsArray);
          } else {
            console.warn('No splits found in API response');
            setSplits([]);
          }
        } else {
          console.error('Failed to fetch splits, status:', response.status);
          const errorText = await response.text();
          console.error('Error response:', errorText);
          setSplits([]);
          notifications.show({
            title: 'Error Fetching Splits',
            message: `Failed to fetch splits for the dataset (Status: ${response.status})`,
            color: 'red',
          });
        }
      } catch (error) {
        console.error('Error fetching splits:', error);
        setSplits([]);
        notifications.show({
          title: 'Error Fetching Splits',
          message: 'An unexpected error occurred while fetching splits',
          color: 'red',
        });
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

  // Update feature options when features change
  useEffect(() => {
    // The 'features' state variable already holds the columns object from the API response
    if (features && typeof features === 'object' && Object.keys(features).length > 0) {
      // Get column names directly from the keys of the 'features' state object
      const columnNames = Object.keys(features);
      const options = columnNames.map((colName: string) => ({ value: colName, label: colName }));
      setFeatureOptions(options);
      console.log('Updated feature options based on features state:', options);
    } else {
      setFeatureOptions([]);
      // Reset dependent selections if features become invalid
      form.setFieldValue('textFields', []);
      form.setFieldValue('labelField', '');
      form.setFieldValue('selectedColumns', []);
      // Log why it was reset (could be initial load, empty response, or error)
      if (features) {
        // Only log if features isn't undefined/null
        console.log(
          'Reset feature options because features object was empty or invalid.',
          features
        );
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [features]); // Keep dependencies, but form is stable

  const handleSelectAllChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { checked } = event.currentTarget;
    setAllColumnsSelected(checked);
    if (checked) {
      form.setFieldValue('selectedColumns', Object.keys(features));
    } else {
      form.setFieldValue('selectedColumns', []);
    }
  };

  const handleSubmit = async (values: FormValues) => {
    setIsSubmitting(true);
    const token = 'YOUR_AUTH_TOKEN_HERE';

    // Determine which columns to actually send (all if checkbox checked)
    const allColumnsSelected =
      !features?.columns ||
      (values.selectedColumns && values.selectedColumns.length === features.columns.length);

    let requestBody: any = {
      name: values.datasetName,
      description: values.description || undefined,
      source: values.dataSource,
      identifier: values.dataSource !== 'HuggingFace' ? values.textFields[0] : values.hfDatasetName,
    };

    if (values.dataSource === 'HuggingFace') {
      requestBody = {
        ...requestBody,
        hf_dataset_name: values.hfDatasetName,
        hf_config: values.hfConfig || undefined,
        hf_split: values.hfSplit || undefined,
        hf_revision: values.hfRevision || undefined,
        hf_token: values.hfToken || undefined,
        text_fields: values.textFields,
        label_field: values.labelField || undefined,
        selected_columns: allColumnsSelected ? undefined : values.selectedColumns,
        limit_rows: values.limitRows,
      };

      // Client-side validation (using Zod automatically handles the textFields check)
      if (!requestBody.hf_dataset_name) throw new Error('HuggingFace Dataset ID is required.');
      if (!requestBody.text_fields || requestBody.text_fields.length === 0)
        throw new Error('Please select at least one Text Field for clustering.');
      if (!allColumnsSelected && (!values.selectedColumns || values.selectedColumns.length === 0)) {
        throw new Error(
          "Please select at least one column to import or check 'Select All Columns'."
        );
      }
    } else {
      // Handle URL/Upload specific fields
      // ... (add validation for URL/Upload if needed)
    }

    try {
      console.log('Sending dataset creation request:', requestBody);

      // Make the API call to create dataset
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.datasets.download}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Include auth token if authentication is implemented
          // 'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        try {
          const result = JSON.parse(errorText);
          throw new Error(result.detail || 'Failed to start dataset import');
        } catch (e) {
          // If parsing as JSON fails, use the raw text
          throw new Error(`Error ${response.status}: ${errorText}`);
        }
      }

      const result = await response.json();
      console.log('Import started:', result);

      // Register the download in our DownloadContext
      if (result.id) {
        addDownload({
          id: result.id,
          name: values.datasetName,
          status: 'pending',
          message: 'Starting download...',
        });

        // Show a toast notification to alert the user about the download
        notifications.show({
          title: 'Dataset Download Started',
          message: 'You can track progress in the download menu in the header',
          color: 'blue',
        });

        // Navigate to datasets list page with tracking info in query params
        const params = new URLSearchParams({
          track_id: result.id.toString(),
          track_name: values.datasetName,
        });
        // Use router.push for navigation
        router.push(`/dashboard/datasets?${params.toString()}`);
      } else {
        // Handle case where ID is missing (shouldn't normally happen on success)
        notifications.show({
          color: 'yellow',
          title: 'Warning',
          message: 'Import started, but could not get ID. Cannot track progress automatically.',
        });
        setIsSubmitting(false); // Ensure spinner stops
      }
    } catch (error: any) {
      console.error('Import error:', error);
      // Check if it's a Zod validation error
      if (error.errors) {
        notifications.show({
          color: 'red',
          title: 'Validation Error',
          message: 'Please check the form fields for errors.',
        });
      } else {
        notifications.show({
          color: 'red',
          title: 'Import Error',
          message: error.message || 'An unexpected error occurred.',
        });
      }
      setIsSubmitting(false);
    } finally {
    }
  };

  // Determine if we should show HuggingFace specific fields
  const showHuggingFaceFields = form.values.dataSource === 'HuggingFace';

  // Prepare column data for MultiSelect
  const columnSelectData = Object.entries(features).map(([name, type]) => ({
    value: name,
    label: `${name} (${type})`,
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
                    placeholder={isLoadingConfigs ? 'Loading...' : 'Select a configuration'}
                    data={configs.map((config) => ({ value: config, label: config }))}
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
                    placeholder={
                      isLoadingSplits
                        ? 'Loading...'
                        : splits.length > 0
                          ? 'Select a split (optional, defaults to all)'
                          : 'No splits available'
                    }
                    data={splits.map((split) => ({ value: split, label: split }))}
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
                  <Group justify="center">
                    <Loader size="sm" /> <Text size="sm">Loading columns...</Text>
                  </Group>
                ) : !form.values.hfDatasetName ? (
                  <Text size="sm" c="dimmed" ta="center">
                    Enter a dataset name above to view available columns
                  </Text>
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
                  <MultiSelect
                    {...form.getInputProps('textFields')}
                    label="Text Field(s) for Clustering"
                    placeholder={
                      isLoadingFeatures
                        ? 'Loading features...'
                        : featureOptions.length > 0
                          ? 'Select one or more text columns'
                          : 'Load features first'
                    }
                    data={featureOptions}
                    disabled={isLoadingFeatures || featureOptions.length === 0}
                    searchable
                    clearable
                    required
                    mt="md"
                    error={form.errors.textFields}
                  />

                  <Select
                    {...form.getInputProps('labelField')}
                    label="Label Field (Optional)"
                    placeholder={
                      isLoadingFeatures
                        ? 'Loading features...'
                        : featureOptions.length > 0
                          ? 'Select a label column'
                          : 'Load features first'
                    }
                    data={featureOptions}
                    disabled={isLoadingFeatures || featureOptions.length === 0}
                    clearable
                    mt="md"
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

            <Button fullWidth mt={30} type="submit" loading={isSubmitting}>
              Create Dataset
            </Button>

            {error && (
              <Text c="red" mt="sm">
                {error}
              </Text>
            )}
          </Paper>
        </Container>
      </form>
    </div>
  );
}
