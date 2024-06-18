import { Center,  Title,  } from '@mantine/core';

export default function Loading() {
    // You can add any UI inside Loading, including a Skeleton.
    return <>
       <Center maw={1000} h={600} >
        
            <Center maw={1000} h={100} >
              <Title
                styles={{
                  root:
                  {
                    "fontSize": 80
                  }
                }}
              >Ivory</Title>

            </Center>
            </Center>
    </>
  }