import { useState } from "react";
import { ChangeEvent, FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { useAppDispatch } from '../redux/hooks';
import { useRegisterMutation } from '../redux/features/authApiSlice';
import { setAuth } from '../redux/features/authSlice';
import { notifications } from '@mantine/notifications';
import classes from "./Error.module.css"
interface FormValues {
  email: string;
  password: string;
}

export default function useRegister(form: { setValue: Function, values: any }) {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const [register, { isLoading }] = useRegisterMutation();
  const [signupError, setSignupError] = useState(false);
  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    form.setValue(name, value);
  };

  const onSubmit = (values: FormValues, event?: React.FormEvent<HTMLFormElement>) => {
    event?.preventDefault();

    const { first_name, last_name, email, password, re_password  } = form.values;

    register({ first_name, last_name, email, password, re_password  })
      .unwrap()
      .then(() => {
        dispatch(setAuth());
        notifications.show({
          title: "Successful",
          message: 'Account created successfully! Please check your email to confirm your account.',
          color: 'teal',
          radius:"md",
          loading: true,
        });
        router.push('/login');
      })
      .catch(() => {
        setSignupError(true);
        notifications.show({
          title: "Error",
          message: 'Something went wrong',
          color: 'red',
          radius:"md",
          classNames:classes
     
        });
      });
  }

  return {
    isLoading,
    onChange,
    onSubmit,
    setValue: form.setValue,
    signupError // Add the setValue function to the return value
    
  };
}