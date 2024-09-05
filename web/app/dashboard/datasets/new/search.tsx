import React, { useState } from 'react';
import { TextInput, ActionIcon, useMantineTheme, rem } from '@mantine/core';
import { IconSearch, IconX } from '@tabler/icons-react';

interface SearchComponentProps {
  onSearch: (searchTerm: string) => void;
  width?: string;        // Optional width prop
  borderRadius?: string;  // Optional borderRadius prop
}

const SearchComponent: React.FC<SearchComponentProps> = ({ onSearch, width = '600px', borderRadius = '8px' }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const theme = useMantineTheme();

  const handleSearch = () => {
    onSearch(searchTerm);
  };

  const handleClear = () => {
    setSearchTerm('');
    onSearch('');
  };

  return (
    <TextInput
      placeholder="Search the dataset"
      value={searchTerm}
      onChange={(event) => setSearchTerm(event.currentTarget.value)}
      radius={borderRadius}
      onKeyPress={(event) => {
        if (event.key === 'Enter') {
          handleSearch();
        }
      }}
      // Apply custom width and borderRadius from props
      style={{ width, borderRadius }}
      leftSection={
        <IconSearch
          style={{ width: rem(16), height: rem(16) }}
          stroke={1.5}
          onClick={handleSearch}
        />
      }
      rightSection={
        searchTerm && (
          <ActionIcon onClick={handleClear} variant="subtle" color="gray">
            <IconX style={{ width: rem(16), height: rem(16) }} stroke={1.5} />
          </ActionIcon>
        )
      }
    />
  );
};

export default SearchComponent;
