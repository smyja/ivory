// Upload.tsx
import React from 'react';
import { FileButton, Button, Group } from '@mantine/core';

interface UploadProps {
  onFilesSelect: (files: File[]) => void;
}

const Upload: React.FC<UploadProps> = ({ onFilesSelect }) => {
  const handleFilesChange = (selectedFiles: File[]) => {
    onFilesSelect(selectedFiles);
  };

  return (
    <Group justify="center">
      <FileButton onChange={handleFilesChange} accept="image/png,image/jpeg" multiple>
        {(props) => <Button {...props}>Upload images</Button>}
      </FileButton>
    </Group>
  );
};

export default Upload;
