import React, { useState } from 'react';
import { TextInput, ActionIcon, useMantineTheme, rem } from '@mantine/core';
import { IconSearch, IconX } from '@tabler/icons-react';

interface SearchComponentProps {
  onSearch: (searchTerm: string) => void;
}

const SearchComponent: React.FC<SearchComponentProps> = ({ onSearch }) => {
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
      onKeyPress={(event) => {
        if (event.key === 'Enter') {
          handleSearch();
        }
      }}
      style={{ width: "600px" }}
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